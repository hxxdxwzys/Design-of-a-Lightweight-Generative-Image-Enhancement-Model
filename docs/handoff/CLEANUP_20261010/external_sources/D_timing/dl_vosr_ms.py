# -*- coding: utf-8 -*-
"""ModelScope 直下 VOSR_0.5B_os (绕过 hf-mirror xet 401)
文件: VOSR_0.5B_os/checkpoints/ema_model.safetensors + VOSR_0.5B_os/args.json
用法: D:/Python/python.exe dl_vosr_ms.py
"""
import os
import sys
import time
from pathlib import Path

import requests

REPO = 'LULALULALU/VOSR_CKPT'
BASE_API = f'https://www.modelscope.cn/api/v1/models/{REPO}/repo'
FILES = ['VOSR_0.5B_os/checkpoints/ema_model.safetensors',
         'VOSR_0.5B_os/args.json']
OUT = Path(r'C:/models/finals/VOSR_ckpts')
CHUNK = 1 << 20


def dl(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dst) + '.part')
    start = tmp.stat().st_size if tmp.exists() else 0
    headers = {'Range': f'bytes={start}-'} if start else {}
    for attempt in range(6):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code in (200, 206):
                    mode = 'ab' if start else 'wb'
                    with open(tmp, mode) as f:
                        for c in r.iter_content(CHUNK):
                            f.write(c)
                    tmp.rename(dst)
                    print(f'[ok] {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)', flush=True)
                    return
                else:
                    print(f'[warn] {url} -> {r.status_code}, retry {attempt}', flush=True)
        except Exception as e:
            print(f'[warn] {e}, retry {attempt}', flush=True)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'failed {url}')


def main():
    for f in FILES:
        dst = OUT / f
        if dst.exists() and dst.stat().st_size > 0:
            print(f'[skip] {f} 已存在', flush=True)
            continue
        url = f'{BASE_API}?FilePath={f}&Revision=master'
        dl(url, dst)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
