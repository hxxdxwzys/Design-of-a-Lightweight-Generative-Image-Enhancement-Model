# -*- coding: utf-8 -*-
"""v5 是否占据"新格子"? 用 LFdiff(低频结构与原图的距离) + 全局统计对比"""
import numpy as np
from pathlib import Path
from PIL import Image
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
DIRS = [('LQ', None, None), ('M24', r'D:/models/final_compliant/output_dir', 43.6951),
        ('TinyV2', r'D:/models/pkg_TinyV2/output_dir', 29.4225),
        ('TinyV2g(同图)', r'D:/models/pkg_TinyV2/output_dir', 33.2872),
        ('v5 (distill)', r'D:/models/pkg_TinyV5d/output_dir', None),
        ('TinyV1', r'D:/models/pkg_TinyV1/output_dir', 27.3521),
        ('sc00', r'D:/models/final_compliant_M24sc00/output_dir', 37.6316)]
IDX = list(range(1, 101, 4)); SCALE = 8
def lf(a):
    h, w = a.shape[:2]
    return np.asarray(Image.fromarray(a).resize((w//SCALE, h//SCALE), Image.BOX), dtype=np.float32)
lqs = {i: lf(np.asarray(Image.open(TEST/f'case{i}.jpg').convert('RGB'), dtype=np.uint8)) for i in IDX}
print(f'{"包":16s} {"LFdiff":>8s} {"lapvar":>8s} {"hf":>6s} {"sat":>6s} {"score":>8s}')
from scipy import ndimage
for name, d, sc in DIRS:
    dif, lv, hf, sat = [], [], [], []
    for i in IDX:
        p = (TEST/f'case{i}.jpg') if d is None else Path(d)/f'case{i}.jpg'
        if not p.exists(): continue
        im = Image.open(p).convert('RGB')
        a = np.asarray(im, dtype=np.uint8)
        dif.append(float(np.abs(lf(a) - lqs[i]).mean()))
        g = np.asarray(im.convert('L'), dtype=np.float32)
        lv.append(float(ndimage.laplace(g).var()))
        hf.append(float(np.sqrt(np.mean((g - ndimage.gaussian_filter(g,2.0))**2))))
        mx = a.max(2).astype(np.float32); mn = a.min(2).astype(np.float32)
        sat.append(float(np.mean((mx-mn)/(mx+1e-6))))
    print(f'{name:16s} {np.mean(dif):8.3f} {np.mean(lv):8.1f} {np.mean(hf):6.2f} {np.mean(sat):6.3f} '
          f'{(str(sc) if sc else "-"):>8s}')
