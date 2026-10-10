# -*- coding: utf-8 -*-
"""标定 torch 版 detail_boost 的剂量 (与 PIL UnsharpMask(r1,p60) 的 2.16x 对齐)"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
sys.path.insert(0, r'D:/tmp_timing')
from detail_boost import detail_boost

SRC = Path(r'D:/models/final_compliant/output_dir')
IDX = [1, 11, 21, 31, 41]

def lapvar(a):
    g = np.asarray(Image.fromarray(a).convert('L'), dtype=np.float32)
    return float(ndimage.laplace(g).var())

for sigma, amount in [(1.0, 0.6), (1.0, 0.45), (1.2, 0.6), (1.5, 0.8)]:
    rs = []
    for i in IDX:
        im = Image.open(SRC / f'case{i}.jpg').convert('RGB')
        a = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        t = torch.from_numpy(a).permute(2, 0, 1)[None]
        o = detail_boost(t, sigma, amount)[0].permute(1, 2, 0).numpy()
        b = np.clip((o + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8)
        rs.append((lapvar(np.asarray(im)), lapvar(b)))
    base = np.mean([r[0] for r in rs]); new = np.mean([r[1] for r in rs])
    print(f'sigma={sigma} amount={amount}: lapvar {base:.1f} -> {new:.1f}  x{new/base:.2f}')
