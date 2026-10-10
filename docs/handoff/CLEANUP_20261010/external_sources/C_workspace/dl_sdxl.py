#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载 SDXL RealVisXL_V4.0 (diffusers 结构, hf-mirror)。

用 huggingface_hub snapshot_download 拉整个目录 (unet/vae/text_encoder 等)。
一次只下这一个 (用户要求)。

用法: python dl_sdxl.py
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download

DEST = 'D:/models/faithdiff_sdxl'
print('开始下载 RealVisXL_V4.0 ->', DEST, flush=True)
path = snapshot_download(
    repo_id='SG161222/RealVisXL_V4.0',
    local_dir=DEST,
    # 只下 FaithDiff 需要的: unet fp16 (variant=fp16), 其他子目录全部
    allow_patterns=['unet/diffusion_pytorch_model.fp16.safetensors', 'unet/config.json',
                    'vae/*', 'text_encoder/*', 'text_encoder_2/*',
                    'tokenizer/*', 'tokenizer_2/*', 'scheduler/*', 'model_index.json'],
)
print('下载完成:', path, flush=True)
# 验证结构
import os as _os
for sub in ['unet', 'vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'scheduler']:
    p = _os.path.join(DEST, sub)
    n = len(_os.listdir(p)) if _os.path.isdir(p) else 0
    print(f'  {sub}: {n} 文件', flush=True)
