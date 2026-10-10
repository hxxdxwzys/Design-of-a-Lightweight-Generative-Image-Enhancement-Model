# -*- coding: utf-8 -*-
"""VOSR 0.5B one-step 权重下载 (hf-mirror, 放 C:/models/finals/VOSR_ckpts)
只下: VOSR_0.5B_os/** + sd21_lwdecoder.pth + torch_cache/** (SD2.1 VAE 本地已有)
用法: D:/Python/python.exe dl_vosr.py
"""
import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.setdefault('HF_HOME', r'C:/models/finals/hf_cache')

from huggingface_hub import snapshot_download  # noqa: E402

ALLOW = ['torch_cache/facebookresearch_dinov2_main/*']


def main():
    p = snapshot_download('CSWRY/VOSR',
                          local_dir=r'C:/models/finals/VOSR_ckpts',
                          allow_patterns=ALLOW,
                          max_workers=4)
    print('[done]', p, flush=True)


if __name__ == '__main__':
    main()
