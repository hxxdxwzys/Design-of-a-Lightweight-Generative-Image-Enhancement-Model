# -*- coding: utf-8 -*-
"""probe_blend: out = LQ + 0.85*(M24 - LQ)  (val 上 PSNR/SSIM/LPIPS 三指标的 Pareto 点)
输出 100 张 4K q98; runner 里做同一条 blend (input 就是 LQ, 天然可得)。
"""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ALPHA = 0.85
M24 = Path(r'D:/models/final_compliant/output_dir')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
DST = Path(r'D:/models/probe_blend/output_dir')
DST.mkdir(parents=True, exist_ok=True)

t0 = time.time()
files = sorted(M24.glob('case*.jpg'), key=lambda p: int(p.stem[4:]))
for n, p in enumerate(files, 1):
    lq = np.asarray(Image.open(TEST / p.name).convert('RGB'), dtype=np.float32)
    pr = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32)
    out = np.clip(lq + ALPHA * (pr - lq), 0, 255).astype(np.uint8)
    Image.fromarray(out).save(DST / p.name, quality=98, subsampling=0)
    if n % 25 == 0:
        print(f'{n}/{len(files)} {time.time()-t0:.0f}s', flush=True)
print('PROBE_BLEND_IMAGES_DONE')
