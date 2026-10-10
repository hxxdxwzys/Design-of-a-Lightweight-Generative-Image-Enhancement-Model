# -*- coding: utf-8 -*-
"""blend 剂量扫描 (val, 有 GT): out = LQ + a*(OUR - LQ)
目的: 找"向保真方向回退"的最优剂量; 这是我们从未测过的轴(有参指标 vs test GT).
PSNR/SSIM 在原生 4K; LPIPS 在中心 512 裁切(CPU 省内存).
"""
import numpy as np
from pathlib import Path
from PIL import Image
import torch

VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
PRED = Path(r'D:/tmp_timing/val_over_sharp')
ALPHAS = [0.0, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]

import lpips
loss_fn = lpips.LPIPS(net='alex', verbose=False)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)

def ssim_np(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = a.mean(); mu2 = b.mean()
    v1 = a.var(); v2 = b.var(); cov = ((a - mu1) * (b - mu2)).mean()
    return ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (v1 + v2 + C2))

print(f'{"alpha":>6} {"PSNR":>7} {"SSIM":>7} {"LPIPS512":>9} {"lapvar":>9}')
res = []
for al in ALPHAS:
    ps, ss, lps, lv = [], [], [], []
    for c in (1, 2, 3):
        lq = np.asarray(Image.open(next(VAL.glob(f'case{c}_lq.*'))).convert('RGB'), dtype=np.float32)
        gt = np.asarray(Image.open(next(VAL.glob(f'case{c}_gt.*'))).convert('RGB'), dtype=np.float32)
        pr = np.asarray(Image.open(PRED / f'case{c}_lq.png').convert('RGB'), dtype=np.float32)
        out = np.clip(lq + al * (pr - lq), 0, 255)
        ps.append(psnr(out, gt)); ss.append(ssim_np(out, gt))
        H, W = out.shape[:2]
        y0, x0 = H // 2 - 256, W // 2 - 256
        t = lambda x: torch.from_numpy(x[y0:y0+512, x0:x0+512].transpose(2, 0, 1)[None] / 127.5 - 1).float()
        with torch.no_grad():
            lps.append(float(loss_fn(t(out), t(gt)).item()))
        from scipy import ndimage
        g = out.mean(axis=2)
        lv.append(float(ndimage.laplace(g).var()))
    res.append((al, np.mean(ps), np.mean(ss), np.mean(lps), np.mean(lv)))
    print(f'{al:6.2f} {np.mean(ps):7.3f} {np.mean(ss):7.4f} {np.mean(lps):9.4f} {np.mean(lv):9.1f}')

# LQ 基线
print('\n[参照] LQ 自身 vs GT:')
a = res[0]
print(f'  alpha=0 就是 LQ: PSNR {a[1]:.3f} SSIM {a[2]:.4f}')
print('BLEND_SWEEP_DONE')
