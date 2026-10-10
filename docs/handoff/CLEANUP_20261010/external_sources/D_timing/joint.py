# -*- coding: utf-8 -*-
"""全 11 次提交联合回归: total = q0 + sum(w*improve) + a*ln(ratio)  (b=0: ratio=1 -> 0)"""
import numpy as np
from itertools import combinations
from scipy.optimize import nnls

LQ = dict(psnr=25.788, ssim=0.7640, lpips=0.4262, dists=0.2942, clipiqa=0.2509, musiq=24.090, maniqa=0.3015, niqe=7.672)
DOWN = {'lpips','dists','niqe'}

OUT = {
 'v3':    dict(psnr=24.883, ssim=0.7574, lpips=0.3869, dists=0.2257, clipiqa=0.3426, musiq=31.490, maniqa=0.3080, niqe=6.560),
 'osediff':dict(psnr=11.071, ssim=0.5660, lpips=0.5670, dists=0.2604, clipiqa=0.2794, musiq=29.908, maniqa=0.2816, niqe=7.236),
 'diffbir':dict(psnr=23.750, ssim=0.5734, lpips=0.6227, dists=0.0, clipiqa=0.5043, musiq=33.525, maniqa=0.3359, niqe=5.179),
 'fusion':dict(psnr=25.136, ssim=0.7566, lpips=0.3817, dists=0.2205, clipiqa=0.3634, musiq=33.955, maniqa=0.3125, niqe=5.812),
 'hypir': dict(psnr=22.127, ssim=0.7293, lpips=0.3364, dists=0.1664, clipiqa=0.4400, musiq=36.814, maniqa=0.3538, niqe=6.215),
 'hydb':  dict(psnr=23.351, ssim=0.7237, lpips=0.3834, dists=0.1785, clipiqa=0.4352, musiq=39.112, maniqa=0.3246, niqe=4.942),
}
R_V3 = 1.28
SUBS = [
    ('v3', R_V3, 'v3', 20.27), ('osediff', 1.0, 'osediff', 10.13), ('diffbir', R_V3, 'diffbir', 18.57),
    ('fusion', R_V3, 'fusion', 20.84), ('fusion', 9.3, 'fusion', 29.85), ('fusion', 16.9, 'fusion', 31.68),
    ('hypir', 16.9, 'hypir', 37.52), ('hydb', 16.9, 'hydb', 37.05),
    ('hypir', 36, 'hypir', 42.21), ('hypir', 87, 'hypir', 47.98), ('hypir', 204, 'hypir', 51.89),
]

def improve(m):
    return np.array([(LQ[k]-m[k]) if k in DOWN else (m[k]-LQ[k]) for k in ['psnr','ssim','lpips','dists','clipiqa','musiq','maniqa','niqe']], dtype=float)

rows = [(name, ratio, out, score) for name, ratio, out, score in SUBS]
results = []
for r in [2,3,4]:
    for cols in combinations(['psnr','ssim','lpips','dists','clipiqa','musiq','maniqa','niqe'], r):
        ci = ['psnr','ssim','lpips','dists','clipiqa','musiq','maniqa','niqe'].index
        # design: [metric improvements, ln(ratio)]
        A = np.array([[improve(OUT[out])[ci(c)] for c in cols] + [np.log(ratio)] for _, ratio, out, _ in rows])
        y = np.array([s for *_, s in rows])
        # nnls with free intercept
        best = None
        for q0 in np.linspace(0, 25, 101):
            w, res = nnls(A, y - q0)
            pred = q0 + A @ w
            errs = pred - y
            mae = np.abs(errs).mean(); maxe = np.abs(errs).max()
            if best is None or (mae, maxe) < (best[0], best[1]):
                best = (mae, maxe, q0, w, errs)
        mae, maxe, q0, w, errs = best
        if mae < 1.0:
            ws = ", ".join(f"{c}={wi:.2f}" for c, wi in zip(cols, w) if wi > 0.001)
            ws += f" | ln_ratio={w[-1]:.2f}"
            results.append((mae, maxe, q0, cols, w, errs))

results.sort(key=lambda t: (t[0], t[1]))
print("=== 联合回归 top (MAE<1.0) ===")
for mae, maxe, q0, cols, w, errs in results[:14]:
    ws = ", ".join(f"{c}={wi:.2f}" for c, wi in zip(cols, w) if wi > 0.001)
    print(f"MAE={mae:.3f} MAX={maxe:.3f} q0={q0:.1f} | {ws}")
    print(f"   residuals: {np.round(errs,2)}")

print("\n=== 速度分量核对 (a*ln(r)) ===")
if results:
    _, _, q0, cols, w, _ = results[0]
    a = w[-1]
    for r in [1.3, 9.3, 16.9, 36, 87, 204]:
        print(f"  {r:>5}x -> {a*np.log(r):+.2f}  (HANDOFF估计: 9.0/10.84/15.53/21.30/25.5)")
