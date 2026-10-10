# -*- coding: utf-8 -*-
"""并行分块下载 Flickr2K.zip —— 针对 us.aws.cdn.hf.co 单流必断的根治方案

原理: 每个 64MB 分块独立发 Range 请求(已验证 CDN 返回 206 + 精确字节),
      掉线只丢一个块, 重试只补该块; 分块落盘 chunk_NNNN.bin, 支持跨进程断点续传。
      全部下完拼成 Flickr2K.zip 并校验 11645004887 字节。
"""
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

URL = os.environ.get('DL_URL', 'https://hf-mirror.com/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip')
TARGET = Path(os.environ.get('DL_TARGET', 'E:/flickr2k/Flickr2K.zip'))
SIZE = int(os.environ.get('DL_SIZE', '11645004887'))
CHUNK = int(os.environ.get('DL_CHUNK', str(64 * 1024 * 1024)))  # 64MB
WORKERS = int(os.environ.get('DL_WORKERS', '4'))
TMP = Path(os.environ.get('DL_TMP', 'E:/flickr2k/chunks'))
UA = {'User-Agent': 'Mozilla/5.0'}


def cpath(i):
    return TMP / f'chunk_{i:04d}.bin'


def fetch(i, start, end):
    """下载 [start,end] 闭区间到 chunk_i, 带指数退避重试"""
    headers = dict(UA)
    headers['Range'] = f'bytes={start}-{end}'
    want = end - start + 1
    for attempt in range(1, 40):
        try:
            r = requests.get(URL, headers=headers, timeout=(30, 300))
            if r.status_code not in (200, 206):
                raise RuntimeError(f'http {r.status_code}')
            data = r.content
            if r.status_code == 200:      # 极少见: 全量返回, 截取区间
                if len(data) < end + 1:
                    raise RuntimeError(f'200 but short {len(data)}')
                data = data[start:end + 1]
            if len(data) != want:
                raise RuntimeError(f'short: got {len(data)} want {want}')
            cpath(i).write_bytes(data)
            return i
        except Exception as e:
            if attempt == 39:
                raise
            time.sleep(min(1.5 * attempt, 20))
    return i


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    n = math.ceil(SIZE / CHUNK)
    jobs = []
    for i in range(n):
        s = i * CHUNK
        e = min(s + CHUNK - 1, SIZE - 1)
        p = cpath(i)
        if p.exists() and p.stat().st_size == (e - s + 1):
            continue                    # 已完成块跳过(跨进程续传)
        jobs.append((i, s, e))
    print(f'[chunks] total={n} remaining={len(jobs)} workers={WORKERS}', flush=True)

    done = 0
    fails = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, i, s, e): (i, s, e) for i, s, e in jobs}
        for fut in as_completed(futs):
            i, s, e = futs[fut]
            try:
                fut.result()
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    print(f'[progress] {done}/{len(jobs)} chunks', flush=True)
            except Exception as ex_:
                fails.append((i, s, e))
                print(f'[fail] chunk {i} (start {s}): {ex_}', flush=True)

    # 串行补跑失败块
    if fails:
        print(f'[retry] {len(fails)} failed chunks', flush=True)
        for i, s, e in fails:
            ok = False
            for attempt in range(20):
                try:
                    fetch(i, s, e)
                    ok = True
                    print(f'[retry-ok] chunk {i}', flush=True)
                    break
                except Exception as ex_:
                    time.sleep(3)
            if not ok:
                print(f'[giveup] chunk {i}: cannot download', flush=True)
                sys.exit(1)

    # 校验所有块齐全
    missing = [i for i in range(n) if not (cpath(i).exists() and
               cpath(i).stat().st_size == (min((i + 1) * CHUNK, SIZE) - i * CHUNK))]
    if missing:
        print(f'[missing] {len(missing)} chunks', flush=True)
        sys.exit(1)

    print('[assemble] writing final file...', flush=True)
    with open(TARGET, 'wb') as out:
        for i in range(n):
            out.write(cpath(i).read_bytes())
    sz = TARGET.stat().st_size
    print(f'[done] {TARGET} size={sz}', flush=True)
    assert sz == SIZE, f'size mismatch {sz} != {SIZE}'
    print('[OK] download complete', flush=True)


if __name__ == '__main__':
    main()
