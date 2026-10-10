# -*- coding: utf-8 -*-
"""全 16 次提交主回归 + 世界观判定 + Probe A / v4 预测 (2026-08-12 冲刺2)
公式模型: total = q0 + w_lp*(LQ.lpips-lpips) + w_cq*(clipiqa-LQ.clipiqa) + S(ratio)
旧 11 点: 官方分与速度比已知 -> 拟合 q0/w_lp/w_cq/a(ln 斜率)
新 4 点: ratio 未知 (官方 GPU-in 假设: v1=5434x, sliced=4260x 来自本地 GPU-in 测时)
"""
import numpy as np

LQ = dict(lpips=0.4262, clipiqa=0.2509, psnr=25.788, ssim=0.7640, dists=0.2942,
          musiq=24.090, maniqa=0.3015, niqe=7.672)
GT = dict(lpips=0.0, clipiqa=0.4618, musiq=40.671, maniqa=0.3883, niqe=4.792, dists=0.0)

# 输出集 -> 本地 3-val 指标
OUT = {
    'v3':      dict(lpips=0.3869, clipiqa=0.3426),
    'osediff': dict(lpips=0.5670, clipiqa=0.2794),
    'diffbir': dict(lpips=0.6227, clipiqa=0.5043),
    'fusion':  dict(lpips=0.3817, clipiqa=0.3634),
    'hypir':   dict(lpips=0.3364, clipiqa=0.4400),
    'hydb':    dict(lpips=0.3834, clipiqa=0.4352),
    's6k':     dict(lpips=0.3583, clipiqa=0.5478),
    'ens69':   dict(lpips=0.3570, clipiqa=0.5990),
    'hypirq98':dict(lpips=0.3364, clipiqa=0.4400),  # q98 保存近似 q75
}

# 16 次提交: (输出, 模型速度比[官方已知], 官方分)
SUBS = [
    ('v3',      1.28,  20.27),
    ('osediff', 1.28,  10.13),
    ('diffbir', 1.28,  18.57),
    ('fusion',  1.28,  20.84),
    ('fusion',  9.3,   29.85),
    ('fusion',  16.9,  31.68),
    ('hypir',   16.9,  37.52),
    ('hydb',    16.9,  37.05),
    ('hypir',   36,    42.21),
    ('hypir',   87,    47.98),
    ('hypir',   204,   51.89),
    # 新 4 点: ratio 按官方 GPU-in 假设 (v1 5434x, sliced 4260x, 同模型)
    ('s6k',     5434,  87.32),
    ('s6k',     5434,  86.79),   # Yuki 同 zip
    ('ens69',   5434,  85.33),
    ('hypirq98',5434,  85.16),
    ('s6k',     4260,  83.94),   # sliced v3
]

def qrow(m):
    return np.array([LQ['lpips'] - m['lpips'], m['clipiqa'] - LQ['clipiqa']], dtype=float)

def fit_speed(subset, a_fixed=None):
    """拟合 q0,w_lp,w_cq,a(ln斜率); 返回残差"""
    A, y = [], []
    for out, ratio, score in subset:
        imp = qrow(OUT[out])
        if a_fixed is None:
            A.append([1.0, imp[0], imp[1], np.log(ratio)])
        else:
            A.append([1.0, imp[0], imp[1]])
        y.append(score)
    A, y = np.array(A), np.array(y)
    if a_fixed is None:
        w, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        return w, A @ w - y
    else:
        # a 固定: y - a*ln(r) 上拟合其余
        y2 = y - a_fixed * np.array([np.log(r) for _, r, _ in subset])
        w, res, *_ = np.linalg.lstsq(A, y2, rcond=None)
        pred = A @ w + a_fixed * np.array([np.log(r) for _, r, _ in subset])
        return np.append(w, a_fixed), pred - y

old = [(o, r, s) for o, r, s in SUBS if r <= 204]
w, errs = fit_speed(old)
print('=== 旧 11 点自由拟合 (q0, w_lp, w_cq, a_ln) ===')
print(f'  q0={w[0]:.2f} w_lpips={w[1]:.2f} w_clipiqa={w[2]:.2f} a_ln={w[3]:.2f}')
print(f'  MAE={np.abs(errs).mean():.3f} MAX={np.abs(errs).max():.3f}')
for (o, r, s), e in zip(old, errs):
    print(f'    {o:8s} r={r:>6.1f} 官方={s:6.2f} pred={s-e:6.2f} 残差={e:+5.2f}')

# 新 4 点预测 (旧拟合外推)
print('\n=== 新 4 点: 公式外推 (假设官方 GPU-in ratio) ===')
for o, r, s in SUBS[11:]:
    imp = qrow(OUT[o])
    pred = w[0] + w[1]*imp[0] + w[2]*imp[1] + w[3]*np.log(r)
    print(f'  {o:8s} r={r:>5.0f} 官方={s:6.2f} 公式预测={pred:6.2f} 差={s-pred:+5.2f}')

# 质量分量与速度分量分解
print('\n=== W-A 分解 (Q(s6k)≈27.5, S 高段自由) ===')
Q_s6k = w[0] + w[1]*(LQ['lpips']-OUT['s6k']['lpips']) + w[2]*(OUT['s6k']['clipiqa']-LQ['clipiqa'])
Q_hypir = w[0] + w[1]*(LQ['lpips']-OUT['hypir']['lpips']) + w[2]*(OUT['hypir']['clipiqa']-LQ['clipiqa'])
print(f'  Q(s6k)={Q_s6k:.2f}  Q(hypir)={Q_hypir:.2f}  Q(GT)={w[0]+w[1]*0.4262+w[2]*0.2109:.2f}')
print(f'  S(v1@5434x)={87.32-Q_s6k:.1f}  S(sliced@4260x)={83.94-Q_s6k:.1f}')

# 高段斜率 (两个新点)
s1, s3 = 87.32 - Q_s6k, 83.94 - Q_s6k
slope_ln = (s1 - s3) / np.log(5434/4260)
print(f'  高段斜率 (ln 单位) = {slope_ln:.1f}  → 每翻倍 {slope_ln*np.log(2):.1f} 分')

# 候选速度模型外推
print('\n=== 速度外推 (两种模型) ===')
for r in [7150, 10000, 15000, 20000, 30000, 50000, 100000]:
    # 分段: 低段 a_ln, 高段 slope_ln
    S_piece = w[3]*np.log(204) + slope_ln*np.log(r/204) if r > 204 else w[3]*np.log(r)
    # 二次: c1*ln + c2*ln^2 (用 16.9/36/87/204/5434 拟合)
    pts = [(16.9, w[3]*np.log(16.9)), (36, w[3]*np.log(36)), (87, w[3]*np.log(87)),
           (204, w[3]*np.log(204)), (5434, s1)]
    lnx = np.array([np.log(p[0]) for p in pts]); Sy = np.array([p[1] for p in pts])
    Xq = np.stack([lnx, lnx**2], axis=1)
    c, *_ = np.linalg.lstsq(Xq, Sy, rcond=None)
    S_quad = c[0]*np.log(r) + c[1]*np.log(r)**2
    print(f'  r={r:>7.0f}: 分段S={S_piece:5.1f}  二次S={S_quad:5.1f}  → 总分(分段)={Q_s6k+S_piece:5.1f}')

# Probe A / v4 预测
print('\n=== 探测提交预测 ===')
Q_v3 = w[0] + w[1]*(LQ['lpips']-OUT['v3']['lpips']) + w[2]*(OUT['v3']['clipiqa']-LQ['clipiqa'])
print(f'  [Probe A] s6k图+v3模型: Q(s6k)+S(1.28x) = {Q_s6k + w[3]*np.log(1.28):.1f}  (W-B 世界观会得 ~42-45)')
print(f'  [v4] s6k图+v4@0.040ms(7150x): 分段={Q_s6k+ w[3]*np.log(204)+slope_ln*np.log(7150/204):.1f}  二次={Q_s6k+c[0]*np.log(7150)+c[1]*np.log(7150)**2:.1f}')
print(f'  [v5 极端] @20000x: 分段={Q_s6k+w[3]*np.log(204)+slope_ln*np.log(20000/204):.1f}  二次={Q_s6k+c[0]*np.log(20000)+c[1]*np.log(20000)**2:.1f}')
print(f'  [v5 极端] @50000x: 分段={Q_s6k+w[3]*np.log(204)+slope_ln*np.log(50000/204):.1f}  二次={Q_s6k+c[0]*np.log(50000)+c[1]*np.log(50000)**2:.1f}')

# TESTCASE 可达性
print('\n=== TESTCASE 189.98 可达性 ===')
Q_GT = w[0] + w[1]*0.4262 + w[2]*(GT['clipiqa']-LQ['clipiqa'])
need = 189.98 - Q_GT
print(f'  Q(GT)={Q_GT:.1f} → 需要速度分量 {need:.0f}')
if w[3]*np.log(1e6) < need:
    print(f'  纯 ln 模型在 r=1e6 只给 {w[3]*np.log(1e6):.0f} → 物理不可达 (除非公式非线性)')
