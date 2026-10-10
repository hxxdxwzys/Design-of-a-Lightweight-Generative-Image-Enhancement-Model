# -*- coding: utf-8 -*-
"""反推官方评分公式: total = Q(metrics) + S(ratio)"""
import numpy as np
from itertools import combinations

LQ = dict(psnr=25.788, ssim=0.7640, lpips=0.4262, dists=0.2942, clipiqa=0.2509, musiq=24.090, maniqa=0.3015, niqe=7.672)
GT = dict(clipiqa=0.4618, musiq=40.671, maniqa=0.3883, niqe=4.792, lpips=0.0, dists=0.0)

# 质量点: (name, official Q, metrics dict)
pts = [
    ("v3",       20.27, dict(psnr=24.883, ssim=0.7574, lpips=0.3869, dists=0.2257, clipiqa=0.3426, musiq=31.490, maniqa=0.3080, niqe=6.560)),
    ("OSEDiff",  10.13, dict(psnr=11.071, ssim=0.5660, lpips=0.5670, dists=0.2604, clipiqa=0.2794, musiq=29.908, maniqa=0.2816, niqe=7.236)),
    ("DiffBIR",  18.57, dict(psnr=23.750, ssim=0.5734, lpips=0.6227, clipiqa=0.5043, musiq=33.525, maniqa=0.3359, niqe=5.179)),  # dists 缺失
    ("fusion",   20.84, dict(psnr=25.136, ssim=0.7566, lpips=0.3817, dists=0.2205, clipiqa=0.3634, musiq=33.955, maniqa=0.3125, niqe=5.812)),
    ("HYPIR",    26.68, dict(psnr=22.127, ssim=0.7293, lpips=0.3364, dists=0.1664, clipiqa=0.4400, musiq=36.814, maniqa=0.3538, niqe=6.215)),
    ("hy_db0.7", 26.21, dict(psnr=23.351, ssim=0.7237, lpips=0.3834, dists=0.1785, clipiqa=0.4352, musiq=39.112, maniqa=0.3246, niqe=4.942)),
]

METRICS = ['psnr','ssim','lpips','dists','clipiqa','musiq','maniqa','niqe']

def build_X(pts, mode):
    X, y = [], []
    for name, q, m in pts:
        row = []
        for k in METRICS:
            if k not in m: row.append(0.0); continue
            if mode == 'delta': row.append(m[k] - LQ[k])
            elif mode == 'raw': row.append(m[k])
            elif mode == 'rel':
                num = m[k] - LQ[k]; den = (GT[k] if k in GT else 1.0) - LQ[k]
                row.append(num/den if den != 0 else 0.0)
        X.append([1.0] + row); y.append(q)
    return np.array(X), np.array(y)

best = []
for mode in ['delta', 'raw', 'rel']:
    X, y = build_X(pts, mode)
    # 尝试不同指标子集 (不含 dists 缺失的 DiffBIR 约束处理: 缺失列置0)
    cand = ['lpips','clipiqa','musiq','maniqa','psnr','ssim','niqe','dists']
    for r in range(1, 7):
        for cols in combinations(cand, r):
            idx = [0] + [METRICS.index(c)+1 for c in cols]  # +1 for intercept
            Xi, yi = X[:, idx], y
            try:
                w, res, rank, sv = np.linalg.lstsq(Xi, yi, rcond=None)
                pred = Xi @ w
                errs = pred - yi
                mae = np.abs(errs).mean(); maxe = np.abs(errs).max()
                if mae < 1.2:
                    best.append((mae, maxe, mode, cols, w, errs))
            except Exception:
                pass

best.sort(key=lambda t: t[0])
print("=== top fits (MAE<1.2) ===")
for mae, maxe, mode, cols, w, errs in best[:12]:
    names = ['intercept'] + list(cols)
    ws = ", ".join(f"{n}={v:+.2f}" for n, v in zip(names, w))
    print(f"[{mode}] MAE={mae:.3f} MAX={maxe:.3f} | {ws}")
    print(f"    residuals: {np.round(errs,2)}")

# 全指标(除dists) delta 拟合用于看整体
X, y = build_X(pts, 'delta')
cols = ['lpips','clipiqa','musiq','maniqa','psnr','ssim','niqe']
idx = [0] + [METRICS.index(c)+1 for c in cols]
w, _, _, _ = np.linalg.lstsq(X[:, idx], y, rcond=None)
pred = X[:, idx] @ w
print("\n=== 全指标 delta 拟合 ===")
for n, v in zip(['intercept']+cols, w): print(f"  {n}: {v:+.3f}")
print("  residuals:", np.round(pred - y, 2))
