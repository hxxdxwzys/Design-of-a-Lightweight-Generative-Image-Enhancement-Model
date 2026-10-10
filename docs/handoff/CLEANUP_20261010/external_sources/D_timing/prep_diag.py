# -*- coding: utf-8 -*-
"""定位特征预处理差异"""
import os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import torch, torch.nn.functional as F
MD = r'D:/models/pkg_PixB/model_dir'
sys.path.insert(0, MD); sys.path.insert(0, r'D:/models/newsrc/PixRestore_code')
from pixrestore.vision import preprocess_dinov2, IMAGENET_MEAN, IMAGENET_STD
from dinov2.models import vision_transformer as _dvt
print('IMAGENET_MEAN =', IMAGENET_MEAN, 'STD =', IMAGENET_STD, flush=True)
x = (torch.rand(1, 3, 512, 512, device='cuda') * 2 - 1)
# repo 路径 (fp32)
a = preprocess_dinov2(x, 448)
print('repo  :', a.dtype, a.shape, float(a.mean()), float(a.std()))
# 我的路径 (fp32 先算)
raw = F.interpolate(x.float().add(1).mul(0.5), size=(448, 448), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1)
mean = x.new_tensor(IMAGENET_MEAN)[None, :, None, None]
std = x.new_tensor(IMAGENET_STD)[None, :, None, None]
b = (raw - mean) / std
print('mine  :', b.dtype, b.shape, float(b.mean()), float(b.std()))
print('maxdiff(preproc) =', float((a - b).abs().max()))
# fp16 路径
c = preprocess_dinov2(x.half(), 448)
print('repo fp16 maxdiff vs fp32 preproc =', float((a - c.float()).abs().max()))
print('PREP_DIAG_DONE')
