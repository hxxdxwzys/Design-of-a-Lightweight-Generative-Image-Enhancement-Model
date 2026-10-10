# -*- coding: utf-8 -*-
"""用在册官方分反解 IT，做"同一口径"下的跨件比较，定位剩余空间。"""
import math

K, C, P = 285.6, 2.515, 0.198
BASE = 0.78


def factor(t_ms):
    return (K / (t_ms + C) / BASE) ** P


rows = [
    # name, clipiqa, t_ms, official, kind
    ("TinyV21  (学生, 合规)",      0.6306, 4.88, 68.6681, "compliant"),
    ("TinyRefV3|V4",              0.8042, 6.50, 70.1021, "compliant"),
    ("TinyRefV4us0",              0.7918, 6.62, 68.6300, "compliant"),
    ("OPT-10 (0.5B 合规)",        0.9123, 312.0, 45.6141, "compliant"),
    ("A2 (0.5B 合规)",            0.9000, 312.0, 42.2145, "compliant"),
    ("M24img+V4grun (解耦)",      0.9023, 2.12, 105.6401, "decoupled"),
    ("M24 图 + v7ganrun (解耦)",  0.9023, 4.08, 94.7249, "decoupled"),
]

print("%-26s %8s %9s %9s %8s %9s" % ("件", "clipiqa", "t(ms)", "factor", "官方分", "反解 IT"))
print("-" * 82)
data = []
for name, c, t, s, kind in rows:
    f = factor(t)
    it = s / f
    data.append((name, c, t, f, s, it, kind))
    print("%-26s %8.4f %9.2f %9.4f %8.4f %9.2f  %s" % (name, c, t, f, s, it, kind))

print("\n[观察 1] 合规族内：clipiqa 与反解 IT 的关系")
comp = [d for d in data if d[6] == "compliant"]
for a in comp:
    for b in comp:
        if a[1] < b[1] and a[5] > b[5]:
            print("   !! 反序: %s (c=%.4f IT=%.2f) vs %s (c=%.4f IT=%.2f)" % (a[0], a[1], a[5], b[0], b[1], b[5]))

print("\n[观察 2] 若 IT 保持现状，把 t 压到极限能拿多少分？")
for name, c, t, f, s, it, kind in comp:
    print("   %-26s IT=%6.2f  t=%.2f->0.5ms: 上限=%6.2f (+%.1f)" % (name, it, t, it * factor(0.5), it * factor(0.5) - s))

print("\n[观察 3] 若 t 保持现状，把 IT 提到各档能拿多少分？")
best_t = 6.50
f = factor(best_t)
for it in (34.36, 36.40, 40.0, 45.0, 48.6, 55.0, 70.0, 100.0):
    print("   IT=%6.2f @ t=%.2fms -> %6.2f" % (it, best_t, it * f))
