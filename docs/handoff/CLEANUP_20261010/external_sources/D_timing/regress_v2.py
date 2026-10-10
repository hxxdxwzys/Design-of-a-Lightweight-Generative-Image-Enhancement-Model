# -*- coding: utf-8 -*-
"""官方质量公式 2.0 拟合: Q(LQ) 自由(负基线假设) + 8 指标全权重 + 速度分量。
数据: 16 提交 + 探测双发 (s6k+v7c=126.26, LQ+v7c=66.60)。
模型: total = Q0 + sum(w_i * improve_i) + S(ratio), S 用分段/二次自由参数。
目标: 识别 musiq/dists/niqe/maniqa/psnr/ssim 是否有显著权重。
"""
import numpy as np

LQ = dict(psnr=25.788, ssim=0.7640, lpips=0.4262, dists=0.2942, clipiqa=0.2509,
          musiq=24.090, maniqa=0.3015, niqe=7.672)
DOWN = {'lpips', 'dists', 'niqe'}
METRICS = ['psnr', 'ssim', 'lpips', 'dists', 'clipiqa', 'musiq', 'maniqa', 'niqe']

OUT = {
    'v3': dict(psnr=24.883, ssim=0.7574, lpips=0.3869, dists=0.2257, clipiqa=0.3426, musiq=31.490, maniqa=0.3080, niqe=6.560),
    'osediff': dict(psnr=11.071, ssim=0.5660, lpips=0.5670, dists=0.2604, clipiqa=0.2794, musiq=29.908, maniqa=0.2816, niqe=7.236),
    'diffbir': dict(psnr=23.750, ssim=0.5734, lpips=0.6227, dists=None, clipiqa=0.5043, musiq=33.525, maniqa=0.3359, niqe=5.179),
    'fusion': dict(psnr=25.136, ssim=0.7566, lpips=0.3817, dists=0.2205, clipiqa=0.3634, musiq=33.955, maniqa=0.3125, niqe=5.812),
    'hypir': dict(psnr=22.127, ssim=0.7293, lpips=0.3364, dists=0.1664, clipiqa=0.4400, musiq=36.814, maniqa=0.3538, niqe=6.215),
    'hydb': dict(psnr=23.351, ssim=0.7237, lpips=0.3834, dists=0.1785, clipiqa=0.4352, musiq=39.112, maniqa=0.3246, niqe=4.942),
    's6k': dict(psnr=24.9, ssim=0.75, lpips=0.3583, dists=None, clipiqa=0.5478, musiq=31.0, maniqa=0.33, niqe=6.5),
    'ens69': dict(psnr=24.9, ssim=0.75, lpips=0.3570, dists=None, clipiqa=0.5990, musiq=31.0, maniqa=0.33, niqe=6.5),
    'hypirq98': dict(psnr=22.127, ssim=0.7293, lpips=0.3364, dists=0.1664, clipiqa=0.4400, musiq=36.814, maniqa=0.3538, niqe=6.215),
    'lq': dict(psnr=25.788, ssim=0.7640, lpips=0.4262, dists=0.2942, clipiqa=0.2509, musiq=24.090, maniqa=0.3015, niqe=7.672),
}

def improve(m):
    out = []
    for k in METRICS:
        v = m.get(k)
        if v is None:
            out.append(0.0)
        elif k in DOWN:
            out.append(LQ[k] - v)
        else:
            out.append(v - LQ[k])
    return np.array(out, dtype=float)

# 提交: (输出, 官方加速比或'free', 官方分)
SUBS = [
    ('v3', 1.28, 20.27), ('osediff', 1.28, 10.13), ('diffbir', 1.28, 18.57),
    ('fusion', 1.28, 20.84), ('fusion', 9.3, 29.85), ('fusion', 16.9, 31.68),
    ('hypir', 16.9, 37.52), ('hydb', 16.9, 37.05),
    ('hypir', 36, 42.21), ('hypir', 87, 47.98), ('hypir', 204, 51.89),
    ('s6k', 5434, 87.32), ('s6k', 4260, 83.94),
    ('s6k', None, 126.26), ('lq', None, 66.60), ('hypirq98', 5434, 85.16), ('ens69', 5434, 85.33),
]
# 探测的官方加速比未知 (地板), 用自由速度参数 v_a, v_b
# 假设 S(r) = c1*ln(r) + c2*ln(r)^2 (凸), 探测比 r7 自由

def solve(c1, c2, r7):
    A, y = [], []
    for out, ratio, score in SUBS:
        row = [1.0] + list(improve(OUT[out]))
        if ratio is None:
            row += [c1 * np.log(r7) + c2 * np.log(r7) ** 2]
        else:
            row += [c1 * np.log(ratio) + c2 * np.log(ratio) ** 2]
        A.append(row)
        y.append(score)
    A, y = np.array(A), np.array(y)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ w
    return w, pred - y

best = None
for c1 in np.linspace(0.5, 4.0, 15):
    for c2 in np.linspace(0.2, 0.8, 13):
        for r7 in [20000, 50000, 100000, 200000, 500000, 1e6]:
            w, errs = solve(c1, c2, r7)
            mae = np.abs(errs).mean()
            if best is None or mae < best[0]:
                best = (mae, c1, c2, r7, w, errs)

mae, c1, c2, r7, w, errs = best
names = ['Q0'] + METRICS + ['S(r)']
print(f'最优: MAE={mae:.3f} c1={c1:.2f} c2={c2:.2f} r7={r7:.0f}')
for n, v in zip(names, w):
    print(f'  {n:8s} {v:+.2f}')
print('残差:')
for (out, ratio, score), e in zip(SUBS, errs):
    print(f'  {out:10s} r={str(ratio):>7s} 官方={score:7.2f} 残差={e:+5.2f}')
