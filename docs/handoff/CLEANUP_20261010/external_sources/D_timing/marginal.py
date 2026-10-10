# -*- coding: utf-8 -*-
"""用【在册官方分】拟合 IT(clipiqa) 曲线，定位边际斜率 —— 回答"质量还值不值得投"。
全部点都是官方读数（score 与各自实测 t），不是代理。
"""
import numpy as np

K, C, P, B = 285.6, 2.515, 0.198, 0.78
f = lambda t: (K / (t + C) / B) ** P

# (标签, clipiqa, t_ms, 官方分, 类别)
PTS = [
    ("TinyV21",        0.6306,   4.88, 68.6681, "micro"),
    ("TinyRefV4us0",   0.7918,   6.62, 68.6300, "micro"),
    ("TinyRefV3|V4",   0.8042,   6.50, 70.1021, "micro"),
    ("OPT-10",         0.9123, 312.00, 45.6141, "teacher"),
    ("A2",             0.9000, 312.00, 42.2145, "teacher"),
]

rows = []
print("%-16s %8s %9s %8s %9s %9s" % ("件", "clipiqa", "t(ms)", "factor", "官方分", "反解 IT"))
for n, c, t, s, k in PTS:
    it = s / f(t)
    rows.append((n, c, t, s, it, k))
    print("%-16s %8.4f %9.2f %8.4f %9.4f %9.3f" % (n, c, t, f(t), s, it))


def fit(points, exclude=()):
    pts = [r for r in points if r[0] not in exclude]
    c = np.array([float(r[1]) for r in pts], dtype=float)
    it = np.array([float(r[4]) for r in pts], dtype=float)
    x = None
    best = None
    for c0 in np.arange(0.0, 0.90, 0.005):
        x = c - c0
        if (x <= 0).any():
            continue
        b, logA = np.polyfit(np.log(x), np.log(it), 1)
        pred = np.exp(logA) * np.power(x, b)
        sse = float(((pred - it) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(c0), float(b), float(np.exp(logA)), pred)
    return best, pts


for label, excl in (("全部 5 点", ()), ("去掉 A2 离群", ("A2",)), ("只用 micro 族 3 点", ("OPT-10", "A2"))):
    best, pts = fit(rows, excl)
    sse, c0, b, A, pred = best
    print("\n--- %s (n=%d) ---" % (label, len(pts)))
    print("  IT = %.3f * (clipiqa - %.3f)^%.4f    SSE=%.3f" % (A, c0, b, sse))
    for r, p in zip(pts, pred):
        print("     %-16s 实测 IT=%7.3f  拟合=%7.3f  残差=%+6.3f (%+.1f%%)"
              % (r[0], r[4], p, p - r[4], (p - r[4]) / r[4] * 100))
    # 边际斜率 at 我们当前的工作点 clipiqa=0.8042
    c_now = 0.8042
    if b != 0 and c_now > c0:
        dIT = A * b * (c_now - c0) ** (b - 1)
        fac = f(6.50)
        print("  ★ 在工作点 clipiqa=0.8042 处：dIT/dc = %.2f  => 每 +0.01 clipiqa = %+.3f 分"
              % (dIT, dIT * 0.01 * fac))
        dIT2 = A * b * (0.90 - c0) ** (b - 1)
        print("    在 clipiqa=0.90 处：dIT/dc = %.2f  => 每 +0.01 clipiqa = %+.3f 分" % (dIT2, dIT2 * 0.01 * fac))

print("\n--- 直接读数（不依赖拟合）：微族内 3 点 ---")
m = [r for r in rows if r[5] == "micro"]
for a in m:
    for b_ in m:
        if a[1] < b_[1]:
            print("  %-15s -> %-15s  d_clipiqa=%+.4f  d_IT=%+.3f  => 每 0.01 clipiqa = %+.3f 分"
                  % (a[0], b_[0], b_[1] - a[1], b_[4] - a[4], (b_[4] - a[4]) / (b_[1] - a[1]) * 0.01 * f(6.5)))

print("\n--- 关键外推：要把 IT 从 33.67 提到 OPT-10 的 44.26，需要多少 clipiqa？---")
best, pts = fit(rows, ("A2",))
sse, c0, b, A, _ = best
for target in (40.0, 44.26, 50.0):
    c_req = c0 + (target / A) ** (1 / b)
    print("  IT=%5.2f  =>  需要 clipiqa = %.4f   (现在 0.8042)" % (target, c_req))
