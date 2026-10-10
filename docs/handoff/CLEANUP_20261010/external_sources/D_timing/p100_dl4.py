# -*- coding: utf-8 -*-
"""P100 pyiqa 权重下载 (2026-09-02): 只下 IQA-PyTorch-Weights (VAE 走本地上传)
用法: python ~/p100/tools/p100_dl4.py
"""
import os
import sys
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'

from huggingface_hub import snapshot_download  # noqa: E402

CACHE = os.path.expanduser('~/.cache/torch/hub')


def main():
    for attempt in range(10):
        try:
            snapshot_download('chaofengc/IQA-PyTorch-Weights',
                              local_dir=os.path.join(CACHE, 'pyiqa'),
                              max_workers=2)
            print('PYIQA OK', flush=True)
            break
        except Exception as e:
            print(f'attempt {attempt + 1} 失败: {type(e).__name__}: {str(e)[:120]}', flush=True)
            time.sleep(60 * (attempt + 1))
    else:
        print('PYIQA_FAIL', flush=True)
        sys.exit(1)
    ck = os.path.join(CACHE, 'pyiqa', 'ckpt_koniq10k.pt')
    print('clipiqa ckpt:', 'OK' if os.path.exists(ck) else 'MISS', flush=True)
    print('PYIQA_DONE', flush=True)


if __name__ == '__main__':
    main()
