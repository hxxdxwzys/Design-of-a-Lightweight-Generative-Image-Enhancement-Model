# -*- coding: utf-8 -*-
"""q_fit.py — 用已知官方分反推的 Q 与本地 native-4K 指标拟合 Q=f(指标).
Q 统一口径: Q = score / ((285.6/(t_ms+2.0))/0.78)**0.19
"""
import json, itertools, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

M = json.loads(Path(r'D:/tmp_timing/q_fit_metrics.json').read_text(encoding='utf-8'))

def Q_of(score, t):
    r = 285.6 / (t + 2.0)
    return score / ((r / 0.78) ** 0.19)

# tag_in_metrics, score, t_local_ms, 备注
ROWS = [
    ('M24',        43.6951, 352.9, '0.5B v7 sc0.8'),
    ('probeA',     43.2837, 352.9, '0.5B v7 sc0.4'),
    ('M24img_v9brun', 94.7249, 4.62, 'M24 图 + v7gan runner(解耦)'),
    ('w2_m0s1_s090', 34.0989, 980.0, '1.4B m0s1 sc0.90'),
    ('w2_m0s7_s097', 31.3942, 980.0, '1.4B m0s7 sc0.97'),
    ('TinyV9b',    40.5757, 23.0, '512 域 0.5M'),
    ('TinyV7gan',  41.7346, 4.62, '256 域 + GAN'),
    ('TinyV1',     27.3521, 46.7, '512 域 1.0M 无 GAN'),
    ('TinyV2',     29.4225, 14.65, '256 域'),
    ('TinyV2g',    33.2872, 5.30, '256 域 + graph'),
    ('TinyV5d',    33.0382, 5.30, '256 域 蒸馏'),
    ('PixB',       18.9710, 158.7, 'PixRestore 像素扩散'),
]
FEATS = ['clipiqa', 'musiq', 'maniqa', 'niqe', 'lapvar']

pts = []
for tag, sc, t, note in ROWS:
    m = M.get(tag)
    if not m:
        print('[miss]', tag); continue
    if any(m.get(f) is None for f in FEATS):
        print('[partial]', tag, {f: m.get(f) for f in FEATS}); continue
    pts.append((tag, Q_of(sc, t), [m[f] for f in FEATS], note))

print('=== 数据点 (Q 统一口径) ===')
print('%-16s %8s | %8s %8s %8s %8s %10s | %s' % ('tag','Q','clipiqa','musiq','maniqa','niqe','lapvar','note'))
for tag, q, x, note in pts:
    print('%-16s %8.2f | %8.4f %8.2f %8.4f %8.4f %10.1f | %s' % (tag, q, x[0], x[1], x[2], x[3], x[4], note))

Y = np.array([p[1] for p in pts]); X = np.array([p[2] for p in pts]); tags=[p[0] for p in pts]
n = len(pts); print('\nn =', n)

def loocv(cols):
    idx = [FEATS.index(c) for c in cols]
    Xs = np.column_stack([X[:, i] for i in idx] + [np.ones(n)])
    errs = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        coef, *_ = np.linalg.lstsq(Xs[keep], Y[keep], rcond=None)
        errs.append(float(Xs[i] @ coef - Y[i]))
    coef, *_ = np.linalg.lstsq(Xs, Y, rcond=None)
    fit = Xs @ coef
    return coef, float(np.sqrt(np.mean(np.array(errs)**2))), float(np.abs(Y-fit).max()), fit

print('\n=== 单指标 ===')
for c in FEATS:
    coef, cv, mx, fit = loocv([c])
    print('  %-8s Q = %+9.4f * %s %+8.2f   LOOCV_rmse=%6.3f  max_res=%6.3f' % (c, coef[0], c, coef[1], cv, mx))
    print('      残差: ' + ' '.join('%s=%+.1f' % (t, y-f) for t, y, f in zip(tags, Y, fit)))

print('\n=== 两指标 (LOOCV 排序) ===')
res2 = []
for c1, c2 in itertools.combinations(FEATS, 2):
    coef, cv, mx, fit = loocv([c1, c2])
    res2.append((cv, mx, c1, c2, coef, fit))
for cv, mx, c1, c2, coef, fit in sorted(res2)[:6]:
    print('  %-8s+%-8s LOOCV_rmse=%6.3f max_res=%6.3f  coef=%s' % (c1, c2, cv, mx, np.round(coef,4)))
best = sorted(res2)[0]
print('  残差(%s+%s): ' % (best[2], best[3]) + ' '.join('%s=%+.1f' % (t, y-f) for t, y, f in zip(tags, Y, best[5])))

print('\n=== 三指标 ===')
res3 = []
for c1, c2, c3 in itertools.combinations(FEATS, 3):
    coef, cv, mx, fit = loocv([c1, c2, c3])
    res3.append((cv, mx, (c1,c2,c3), coef, fit))
for cv, mx, cs, coef, fit in sorted(res3)[:4]:
    print('  %-24s LOOCV_rmse=%6.3f max_res=%6.3f coef=%s' % ('+'.join(cs), cv, mx, np.round(coef,4)))

print('\n=== 全场最小二乘 (5 特征, 只作参考, 无泛化意义) ===')
coef, cv, mx, fit = loocv(FEATS)
print('  coef=', np.round(coef, 4), ' LOOCV_rmse=%.3f' % cv)

print('\n=== 外推: GT(val) / LQ(test) 的预测 Q ===')
for extra in ('GT_val', 'LQ'):
    m = M.get(extra)
    if not m: continue
    x = np.array([m[f] for f in FEATS] + [1.0])
    line = '%-8s metrics: ' % extra + ' '.join('%s=%.4g' % (f, m[f]) for f in FEATS)
    print(line)
    for cv, mx, cs, coef, fit in sorted(res2)[:3]:
        idx = [FEATS.index(c) for c in cs]
        c2_, *_ = np.linalg.lstsq(np.column_stack([X[:, i] for i in idx] + [np.ones(n)]), Y, rcond=None)
        print('    %-18s -> Q_pred = %.1f' % ('+'.join(cs), float(np.array([m[f] for f in cs] + [1.0]) @ c2_)))
    coef1, *_ = np.linalg.lstsq(np.column_stack([X[:,0], np.ones(n)]), Y, rcond=None)
    print('    clipiqa 单指标      -> Q_pred = %.1f' % float(np.array([m['clipiqa'],1.0]) @ coef1))
print('Q_FIT_DONE')
