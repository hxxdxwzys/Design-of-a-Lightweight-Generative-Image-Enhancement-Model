# -*- coding: utf-8 -*-
"""统一口径对照: M24 vs PixRestore-b/l/xl  (val 512 中心裁切)
M24 的 val 出图在 D:/tmp_timing/val_over_sharp/caseN_lq.png
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np, torch
from pathlib import Path
from PIL import Image
import lpips as lpips_mod

VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
M24V = Path(r'D:/tmp_timing/val_over_sharp')
lp = lpips_mod.LPIPS(net='alex', verbose=False)
def psnr(p, g):
    m = np.mean((p.astype(np.float64) - g.astype(np.float64)) ** 2)
    return 99.0 if m <= 0 else 10 * np.log10(255.0 ** 2 / m)
def ssim_np(p, g):
    p = p.astype(np.float64); g = g.astype(np.float64)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    m1, m2 = p.mean(), g.mean(); v1, v2 = p.var(), g.var()
    cov = ((p-m1)*(g-m2)).mean()
    return ((2*m1*m2+C1)*(2*cov+C2))/((m1**2+m2**2+C1)*(v1+v2+C2))
def metrics(pred512, gt512):
    t = lambda z: torch.from_numpy(z.transpose(2,0,1)[None]/127.5-1).float()
    with torch.no_grad():
        return psnr(pred512, gt512), ssim_np(pred512, gt512), float(lp(t(pred512), t(gt512)).item())
rows = {}
for c in (1, 2, 3):
    lqim = Image.open(next(VAL.glob(f'case{c}_lq.*'))).convert('RGB')
    gtim = Image.open(next(VAL.glob(f'case{c}_gt.*'))).convert('RGB')
    W, H = lqim.size
    box = ((W-512)//2, (H-512)//2, (W-512)//2+512, (H-512)//2+512)
    gt = np.asarray(gtim.crop(box), dtype=np.uint8)
    lq = np.asarray(lqim.crop(box), dtype=np.uint8)
    m24p = M24V / f'case{c}_lq.png'
    m24 = np.asarray(Image.open(m24p).convert('RGB').crop(box), dtype=np.uint8) if m24p.exists() else None
    for name, arr in (('LQ', lq), ('M24', m24)):
        if arr is None: continue
        rows.setdefault(name, []).append(metrics(arr, gt))
for k, v in rows.items():
    a = np.array(v)
    print(f'{k:14s} PSNR {a[:,0].mean():6.3f} | SSIM {a[:,1].mean():.4f} | LPIPS {a[:,2].mean():.4f}')
print('(同口径: 512 中心裁切 = 前面 PixRestore-b: PSNR 23.709 / SSIM 0.9199 / LPIPS 0.4107)')
print('REF_DONE')
