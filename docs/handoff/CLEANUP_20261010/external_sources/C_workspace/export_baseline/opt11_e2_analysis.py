r"""E2 分析：官方 IT 与各代理的回归（旧时代 9 点，含 M24/OPT10 高锚点）。"""
import json, math, statistics
from pathlib import Path
P = Path(r"D:\models\opt11")
# tag -> (官方分, t_ms)
OFF = {
 "TinyV1":   (27.3521, 44.83),
 "TinyV2":   (29.4225, 12.65),
 "TinyV2g":  (33.2872,  3.30),
 "TinyV5d":  (33.0382,  3.30),
 "TinyV7gan":(41.7346,  2.61),
 "TinyV9b":  (40.5757, 21.03),
 "PixB":     (18.9710, 155.9),
 "M24img":   (43.6951, 366.2),
 "OPT10img": (45.6141, 324.0),
}
BASE, C0, p = 285.6, 2.0, 0.19
factor = lambda t: (BASE/(t+C0)/0.78)**p
rows = []
for tag, (s, t) in OFF.items():
    f = P/("proxy_%s.json" % tag)
    if not f.exists(): print("missing proxy:", tag); continue
    d = json.loads(f.read_text(encoding="utf-8"))
    rows.append({"tag": tag, "IT": s/factor(t), **{k: d[k] for k in ("clipiqa","niqe","musiq","maniqa","lapvar")}})
print("== 数据 ==")
print("%-10s %7s %8s %8s %8s %8s %8s" % ("tag","IT","clipiqa","niqe","musiq","maniqa","lapvar"))
for r in rows:
    print("%-10s %7.2f %8.4f %8.4f %8.2f %8.4f %8.1f" % (r["tag"], r["IT"], r["clipiqa"], r["niqe"], r["musiq"], r["maniqa"], r["lapvar"]))

def fit_power(xs, ys):
    lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    sxx = sum((x-mx)**2 for x in lx); sxy = sum((x-mx)*(y-my) for x, y in zip(lx, ly))
    b = sxy/sxx; lA = my - b*mx
    ss_tot = sum((y-my)**2 for y in ly); ss_res = sum((y-(lA+b*x))**2 for x, y in zip(lx, ly))
    return math.exp(lA), b, 1-ss_res/ss_tot

def fit_linear(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    b = sxy/sxx; a = my - b*mx
    ss_tot = sum((y-my)**2 for y in ys); ss_res = sum((y-(a+b*x))**2 for x, y in zip(xs, ys))
    return a, b, 1-ss_res/ss_tot

print()
print("== 单代理回归（幂律 IT=A·x^b / 线性 IT=a+b·x）==")
for key in ("clipiqa","niqe","musiq","maniqa","lapvar"):
    xs = [r[key] for r in rows]; ys = [r["IT"] for r in rows]
    A, b, r2p = fit_power(xs, ys)
    a, bb, r2l = fit_linear(xs, ys)
    print("   %-8s 幂律 A=%8.2f b=%6.3f R2=%.3f | 线性 a=%8.2f b=%8.3f R2=%.3f" % (key, A, b, r2p, a, bb, r2l))

print()
print("== 组合代理（乘积型 IT=A·x^b·y^c，二元最小二乘）==")
import itertools
best = []
for k1, k2 in itertools.combinations(("clipiqa","musiq","maniqa","lapvar"), 2):
    X = [[1.0, math.log(r[k1]), math.log(r[k2])] for r in rows]
    Y = [math.log(r["IT"]) for r in rows]
    # 正规方程 3x3
    import numpy as np
    Xa = np.array(X); Ya = np.array(Y)
    coef, res, rank, sv = np.linalg.lstsq(Xa, Ya, rcond=None)
    pred = Xa @ coef
    ss_tot = float(((Ya - Ya.mean())**2).sum()); ss_res = float(((Ya - pred)**2).sum())
    r2 = 1 - ss_res/ss_tot
    best.append((r2, k1, k2, coef))
    print("   %-8s × %-8s : R2=%.4f  A=%.2f b1=%.3f b2=%.3f" % (k1, k2, r2, math.exp(coef[0]), coef[1], coef[2]))
best.sort(reverse=True)
print()
print("   ⇒ 最佳组合：%s × %s (R2=%.4f)" % (best[0][1], best[0][2], best[0][0]))
print("E2_ANALYSIS_DONE")
