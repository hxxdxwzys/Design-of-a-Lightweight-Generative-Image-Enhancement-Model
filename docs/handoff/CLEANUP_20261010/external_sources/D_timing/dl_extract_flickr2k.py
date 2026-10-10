# -*- coding: utf-8 -*-
"""稳健下载 Flickr2K.zip (huggingface_hub, 断点续传) + 解压到 D:/datasets/flickr2k

为什么换掉 curl:
  - curl 走 hf-mirror 到 us.aws.cdn.hf.co (Xet 存储), 下载到 ~95% 掉线后 Range 续传失效,
    每次都从头来 (已在 11.1GB 处白跑一轮)。
  - huggingface_hub 自带 .incomplete 分块 + Range 续传 + 指数退避重试, 掉线只补缺块。
用法: python dl_extract_flickr2k.py
"""
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 缓存放 E 盘 (默认在 C: 用户目录, 11.6G 会撑爆 C)
os.environ['HF_HOME'] = 'E:/hf_cache'
# 禁用 hf_transfer 多线程 (对不稳网络反而易出错), 走标准 http + 续传
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from huggingface_hub import hf_hub_download  # noqa: E402

ZIP = Path('E:/flickr2k/Flickr2K.zip')
OUT = Path('D:/datasets/flickr2k')
TARGET_SIZE = 11645004887
EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp'}


def download():
    last_err = None
    for attempt in range(1, 500):
        try:
            p = hf_hub_download(
                repo_id='yangtao9009/Flickr2K',
                filename='Flickr2K.zip',
                repo_type='dataset',
                local_dir=str(ZIP.parent),
                resume_download=True,
            )
            print(f'[dl] attempt {attempt}: OK -> {p}', flush=True)
            return
        except Exception as e:
            last_err = e
            print(f'[dl] attempt {attempt}: {type(e).__name__}: {e}', flush=True)
            msg = str(e)
            if '404' in msg or 'Repository Not Found' in msg or 'Entry Not Found' in msg:
                raise  # 永久错误, 不重试
            time.sleep(20)
    raise RuntimeError(f'download gave up: {last_err}')


def verify_and_extract():
    sz = ZIP.stat().st_size
    print(f'[verify] size={sz}', flush=True)
    if sz != TARGET_SIZE:
        raise RuntimeError(f'size mismatch {sz} != {TARGET_SIZE}')
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(ZIP) as z:
        for nm in z.namelist():
            if nm.endswith('/'):
                continue
            ext = Path(nm).suffix.lower()
            if ext not in EXT:
                continue
            try:
                (OUT / f'flickr_{n:04d}{ext}').write_bytes(z.read(nm))
                n += 1
                if n % 500 == 0:
                    print(f'[extract] ... {n}', flush=True)
            except Exception as e:
                print(f'[extract] skip {nm}: {e}', flush=True)
    print(f'[extract] done: {n} images -> {OUT}', flush=True)


if __name__ == '__main__':
    download()
    verify_and_extract()
    print('[ALL_DONE]', flush=True)
