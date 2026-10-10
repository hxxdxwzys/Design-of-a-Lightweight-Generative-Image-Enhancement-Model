# -*- coding: utf-8 -*-
"""q_fit2.py — 用修正后的分数模型 (Q=43.43, p=0.198, C=2.515) 反推各图像集的 Q, 拟合 Q=f(指标).
Q = score / ((285.6/(t+2.515))/0.78)**0.198     [由 4 个同图点自由拟合得到, 残差 ±0.01]
"""
import json, itertools, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
M = json.loads(Path(r'D:/tmp_timing/q_fit_metrics2.json').read_text(encoding='utf-8')) if Path(r'D:/tmp_timing/q_fit_metrics2.json').exists() else {}
MUSIQ = {'M24': 46.1257, 'w2_m0s1_s090': 38.629, 'w2_m0s7_s097': 37.8934}   # 原生 4K, 30 张, 历史实测
P, C = 0.198, 2.515
def Q_of(score, t):
    return score / ((285.6 / (t + C)) / 0.78) ** P
ROWS = [
    ('M24',            43.6951, 352.9, '0.5B v7 sc0.8 (我们的图)'),
    ('probeA',         43.2837, 352.9, '0.5B v7 sc0.4'),
    ('w2_m0s1_s090',   34.0989, 980.0, '1.4B m0s1 sc0.90'),
    ('w2_m0s7_s097',   31.3942, 980.0, '1.4B m0s7 sc0.97'),
    ('TinyV9b',        40.5757,  33.7, '512 域 0.38M 学生'),
    ('TinyV7gan',      41.7346,   4.62, '256 域 + GAN'),
    ('TinyV2g',        33.2872,   5.30, '256 域 + graph'),
    ('TinyV5d',        33.0382,   5.30, '256 域 蒸馏'),
    ('TinyV1',         27.3521,  46.7, '512 域 1.0M'),
    ('TinyV2',         29.4225,  14.65, '256 域'),
    ('PixB',           18.9710, 158.7, 'PixRestore 像素扩散'),
]
FE = ['maniqa', 'clipiqa', 'niqe', 'lapvar']
pts = []
for tag, sc, t, note in ROWS:
    m = M.get(tag)
    if not m or m.get('n', 0) == 0 or any(m.get(f) is None for f in FE):
        print('[miss]', tag, (m or {}).get('n')); continue
    pts.append((tag, Q_of(sc, t), [m[f] for f in FE], note))
n = len(pts)
print('=== 数据点 (Q 修正口径 p=0.198 C=2.515) ===')
print('%-14s %7s | %8s %8s %8s %9s | %s' % ('tag','Q','maniqa','clipiqa','niqe','lapvar','note'))
for tag, q, x, note in pts:
    print('%-14s %7.2f | %8.4f %8.4f %8.4f %9.1f | %s' % (tag, q, x[0], x[1], x[2], x[3], note))
Y = np.array([p[1] for p in pts]); X = np.array([p[2] for p in pts]); tags = [p[0] for p in pts]
print('\nn =', n)
def loocv(cols):
    idx = [FE.index(c) for c in cols]
    Xs = np.column_stack([X[:, i] for i in idx] + [np.ones(n)])
    errs = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        coef, *_ = np.linalg.lstsq(Xs[keep], Y[keep], rcond=None)
        errs.append(float(Xs[i] @ coef - Y[i]))
    coef, *_ = np.linalg.lstsq(Xs, Y, rcond=None)
    fit = Xs @ coef
    return coef, float(np.sqrt(np.mean(np.array(errs) ** 2))), fit
print('=== 单指标 (LOOCV) ===')
for c in FE:
    coef, cv, fit = loocv([c])
    print('  %-8s Q = %+10.4f*%s %+8.2f   LOOCV=%.2f  残差: %s' % (
        c, coef[0], c, coef[1], cv, ' '.join('%s=%+.1f' % (t, y-f) for t, y, f in zip(tags, Y, fit))))
print('=== 两指标 (LOOCV 排序) ===')
res = []
for c1, c2 in itertools.combinations(FE, 2):
    coef, cv, fit = loocv([c1, c2]); res.append((cv, c1, c2, coef, fit))
for cv, c1, c2, coef, fit in sorted(res)[:3]:
    print('  %s+%s  LOOCV=%.2f  coef=%s' % (c1, c2, cv, np.round(coef, 4)))
    print('     残差: ' + ' '.join('%s=%+.1f' % (t, y-f) for t, y, f in zip(tags, Y, fit)))
print('\n=== 外推检查: GT(val) / LQ(test) / 我们最好的图 ===')
for extra in ('GT_val', 'LQ'):
    m = M.get(extra)
    if not m or m.get('n', 0) == 0: continue
    print('  %-8s maniqa=%.4f clipiqa=%.4f niqe=%.4f lapvar=%.1f' % (extra, m['maniqa'], m['clipiqa'], m['niqe'], m['lapvar']))
    for cv, c1, c2, coef, fit in sorted(res)[:2]:
        v = coef[0] * m[c1] + coef[1] * m[c2] + coef[2]
        print('     %s+%s -> Q_pred=%.1f' % (c1, c2, v))
    coef1, _, _ = loocv(['maniqa'])
    print('     maniqa 单指标 -> Q_pred=%.1f' % (coef1[0] * m['maniqa'] + coef1[1]))
print('Q_FIT2_DONE')
