# -*- coding: utf-8 -*-
"""检查 (Q,p,C) 的可辨识性: 同一批图(M24)在不同时延下的 4 个官方读数."""
import numpy as np, sys
from scipy.optimize import least_squares
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# t_local_ms, official_score   (图逐字节/逐像素相同: M24, Slow8, M24img+V7ganrun, M24img+V4grun)
PTS = [(352.9, 43.6951), (2930.0, 28.7538), (4.62, 94.7249), (1.60, 105.6401)]
BASE = 285.6

def model(params, t):
    Q, p, C = params
    r = BASE / (t + C)
    return Q * (r / 0.78) ** p

def resid(params):
    return np.array([model(params, t) - s for t, s in PTS])

best = None
for C0 in (0.01, 1.0, 2.0, 5.0, 20.0):
    for p0 in (0.1, 0.19, 0.3):
        try:
            r = least_squares(resid, [43.0, p0, C0], bounds=([1, 0.01, 1e-6], [500, 2.0, 500.0]))
            sse = float((r.fun**2).sum())
            if best is None or sse < best[0]:
                best = (sse, r.x)
        except Exception as e:
            pass
sse, (Q, p, C) = best
print('=== 无约束最优 (Q,p,C) = (%.3f, %.4f, %.4f)  SSE=%.4f  max|res|=%.3f' % (Q, p, C, sse, max(abs(resid([Q,p,C])))))
for t, s in PTS:
    print('   t=%8.2f  官方 %8.4f  预测 %8.4f  %+.3f' % (t, s, model([Q,p,C], t), model([Q,p,C], t)-s))
print('   -> 官方加速比上限 (t->0) = %.1fx   M24 图在此上限 = %.1f 分' % (BASE/C, model([Q,p,C], 0.0)))

print('\n=== 固定不同 C 时的最优 (Q,p) 与残差 ===')
for Cf in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
    def resid2(params):
        Q2, p2 = params
        return np.array([Q2 * ((BASE/(t+Cf))/0.78)**p2 - s for t, s in PTS])
    r = least_squares(resid2, [43.0, 0.19], bounds=([1, 0.01], [500, 2.0]))
    sse = float((r.fun**2).sum())
    Q2, p2 = r.x
    print('  C=%5.1f -> Q=%7.3f p=%.4f  r_max=%7.1fx  SSE=%8.4f  max_res=%.3f   M24上限=%.1f' % (
        Cf, Q2, p2, (BASE/Cf if Cf > 0 else float("inf")), sse, max(abs(r.fun)), Q2*((BASE/Cf)/0.78)**p2 if Cf > 0 else float('nan')))

print('\n=== 若榜首 217.21 用同样这批图 (Q=%.2f), 需要 r 多大? ===' % Q)
for sc in (217.21, 157.06, 124.35, 121.79, 109.79):
    need = (sc / Q) ** (1.0/p) * 0.78
    print('  score %7.2f -> r_official %10.1f -> t_official %8.4f ms' % (sc, need, BASE/need))
print('CEILING_CHECK_DONE')
