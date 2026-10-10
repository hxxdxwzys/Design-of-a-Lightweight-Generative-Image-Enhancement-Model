# -*- coding: utf-8 -*-
"""score_pred2.py (2026-09-15 晚) —— 分数模型 v3（乘法幂律）唯一判分工具。

  预测分  = Q(图) * (r_official/0.78)^0.19
  r_official = 285.6 / (t_local_ms + 2.0)          # 官方口径含 ~2ms 固定开销（实测反解）
  天花板  = 285.6/2.0 = 142.8x                      # t->0 的极限
  8 点自验误差 ±5%（见 HANDOFF 22 / 对应表 一·G）

用法:
  python score_pred2.py pred  <Q> <t_ms>      # 给图像质量系数 Q 与本地时延，算预测分
  python score_pred2.py invert <score> <t_ms> # 用已知官方分反推这批图的 Q
  python score_pred2.py table                 # 打印已知锚点与天花板
注意: 旧工具 score_pred.py 是【已证伪】的加法模型(IT+S)，不要再用于预测。
"""
import math, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
P = 0.19
C = 2.0
BASE = 285.6


def r_of(t_ms):
    return BASE / (t_ms + C)


def score(Q, t_ms):
    return Q * (r_of(t_ms) / 0.78) ** P


def Q_of(score_val, t_ms):
    return score_val / (r_of(t_ms) / 0.78) ** P


ANCHORS = [
    ('M24 v3（合规，图+runner 同主体）', 43.6951, 0.8, 352.9, 43.70),
    ('M24avg2（同图，2 样本平均）', 36.6414, 0.0, 748.5, 43.70),
    ('ProbeSlow8（同图，8x 慢）', 28.7538, 0.0, 2930.0, 43.70),
    ('TinyV2 / TinyV2g（同图两个速度）', 29.4225, 0.0, 14.65, 15.5),
    ('TinyV9b（合规学生）', 40.5757, 0.0, 22.4, 23.4),
    ('M24img+V7ganrun（解耦）', 94.7249, 0.0, 4.62, 43.70),
    ('M24img+V4grun（解耦）', 105.6401, 0.0, 1.60, 43.70),
]

if len(sys.argv) < 2:
    print(__doc__); sys.exit(0)
mode = sys.argv[1]
if mode == 'pred':
    Q, t = float(sys.argv[2]), float(sys.argv[3])
    print('Q=%.2f  t=%.2f ms -> r_official=%.1fx -> 预测分 %.2f' % (Q, t, r_of(t), score(Q, t)))
elif mode == 'invert':
    s, t = float(sys.argv[2]), float(sys.argv[3])
    print('官方分 %.4f  t=%.2f ms -> r_official=%.1fx -> 反推 Q=%.2f' % (s, t, r_of(t), Q_of(s, t)))
else:
    print('%-34s %10s %9s %8s %8s %9s' % ('提交', '官方分', '本地ms', 'r_official', '模型预测', '偏差'))
    for name, s, _f, t, Q in ANCHORS:
        pr = score(Q, t)
        print('%-34s %10.4f %9.2f %8.1f %8.2f %+9.2f' % (name, s, t, r_of(t), pr, pr - s))
    print()
    print('官方加速比天花板 (t->0): %.1fx  => M24 的图在此上限处也只能拿到 %.1f 分' % (r_of(0.0), score(43.70, 0.0)))
    print('榜首 217 反推需要的 Q: %.1f (同口径)' % Q_of(217.21, 0.5))
