# -*- coding: utf-8 -*-
"""用新的 105.6401 反解: 官方时延口径是否含固定开销 + 幂指数 p"""
import sys, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
Q_M24 = 43.6951          # M24 自己那一发(0.78x) => Q(图)=43.6951
obs = {'v7gan_runner': 94.7249, 'v4g_runner': 105.6401}
t_loc = {'v7gan_runner': 4.62, 'v4g_runner': 1.60}   # 本地空闲 median(v1 口径)
print('两发同图(M24) 只换 runner: 94.7249 -> 105.6401  (x%.4f)' % (obs['v4g_runner'] / obs['v7gan_runner']))
r = obs['v4g_runner'] / obs['v7gan_runner']
print()
print('%-8s %8s %10s %10s %10s' % ('c(ms)', 'p', 'r_v7gan', 'r_v4g', 'check_v7gan'))
for c in [0.0, 1.0, 2.0, 2.04, 3.0, 4.0]:
    r1 = 285.6 / (t_loc['v7gan_runner'] + c)
    r2 = 285.6 / (t_loc['v4g_runner'] + c)
    if abs(math.log(r2 / r1)) < 1e-9:
        continue
    p = math.log(r) / math.log(r2 / r1)
    pred = Q_M24 * (r1 / 0.78) ** p
    print('%-8.2f %8.4f %10.1f %10.1f %10.2f' % (c, p, r1, r2, pred))
print()
print('=== 用 c=2.0, p=0.19 复验全部同图/合规读数 ===')
c, p = 2.0, 0.19
def ratio(t): return 285.6 / (t + c)
def sc(Q, t): return Q * (ratio(t) / 0.78) ** p
print('  M24   (Q43.70, t352.9) -> %.2f  (实测 43.6951)' % sc(43.6951, 352.9))
print('  avg2  (Q43.70, t748.5) -> %.2f  (实测 36.6414)' % sc(43.6951, 748.5))
print('  Slow8 (Q43.70, t2930)  -> %.2f  (实测 28.7538)' % sc(43.6951, 2930))
print('  V2    (Q15.5,  t14.65) -> %.2f  (实测 29.4225)' % sc(15.5, 14.65))
print('  V2g   (Q15.5,  t5.30)  -> %.2f  (实测 33.2872)' % sc(15.5, 5.30))
print('  v9b   (Q23.4,  t22.4)  -> %.2f  (实测 40.5757)' % sc(23.4, 22.4))
print()
print('=== 官方加速比的天花板 (c=2.0) ===')
for t in [0.0, 0.5, 1.0, 1.6, 2.0, 4.62]:
    print('  本地 %.2f ms -> 官方 %.1fx' % (t, ratio(t)))
print('  => 理论上限 = 285.6/2.0 = %.0fx (再快也超不过)' % ratio(0))
print('  => M24 的图在理论上限处的分 = %.1f' % sc(43.6951, 0.0))
