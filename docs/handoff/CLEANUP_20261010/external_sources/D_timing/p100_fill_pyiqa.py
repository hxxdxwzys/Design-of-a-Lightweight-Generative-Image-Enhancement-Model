# -*- coding: utf-8 -*-
"""找出 pyiqa 缓存缺失文件并用直连 resolve URL 补齐 (2026-09-02)"""
import os
import subprocess

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import list_repo_files  # noqa: E402

CACHE = os.path.expanduser('~/.cache/torch/hub/pyiqa')


def main():
    tok = open(os.path.expanduser('~/p100/hf_token')).read().strip()
    files = list_repo_files('chaofengc/IQA-PyTorch-Weights', token=tok)
    have = set(os.listdir(CACHE))
    missing = [f for f in files if f not in have]
    print(f'repo 共 {len(files)} 文件, 本地 {len(have)} 个, 缺 {len(missing)} 个', flush=True)
    for f in missing:
        print('MISS:', f, flush=True)
    # 用 resolve 直连下载 (绕开 API 限流)
    for f in missing:
        url = f'https://hf-mirror.com/chaofengc/IQA-PyTorch-Weights/resolve/main/{f}'
        dst = os.path.join(CACHE, f)
        r = subprocess.run(['curl', '-sL', '-m', '600', '-o', dst, url])
        size = os.path.getsize(dst) if os.path.exists(dst) else 0
        print(f'  -> {f}: {size} bytes (rc={r.returncode})', flush=True)
    print('FILL_DONE', flush=True)


if __name__ == '__main__':
    main()
