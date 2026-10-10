r"""E1: 分数模型全量重拟与自洽性检验（纯 CPU）。
数据来源：docs/reference/新世界模型分数对应表.md / 规则细则与判例台账.md / 2026-09-20 官方 5 发。
"""
import math, itertools, statistics

D = [
    ("M24_v3",        43.6951, 0.9023, 366.2, "old", "m24",  "本体"),
    ("M24_v4",        43.5844, 0.9023, 366.2, "old", "m24",  "同图"),
    ("M24img_V7gan",  94.7249, 0.9023,   4.62, "old", "m24",  "同图+快测速件"),
    ("M24img_V4g",   105.6401, 0.9023,   1.60, "old", "m24",  "同图+更快测速件"),
    ("ProbeSlow8",    28.7538, 0.9023, 2929.6, "old", "m24",  "同图 x8 时延"),
    ("probeA_sc04",   43.2837, 0.7700, 366.2, "old", "m24",  "LoRA 减半"),
    ("M24sc00",       37.6316, 0.6448, 366.2, "old", "m24",  "去 LoRA"),
    ("TinyV1",        27.3521, 0.3367,  44.83, "old", "tiny", "r=6.1"),
    ("TinyV2",        29.4225, 0.3532,  12.65, "old", "tiny", "r=19.5"),
    ("TinyV2g",       33.2872, 0.3532,   3.30, "old", "tiny", "r=53.9"),
    ("TinyV5d",       33.0382, 0.3897,   3.30, "old", "tiny", "r=53.9"),
    ("TinyV7gan",     41.7346, 0.5059,   2.61, "old", "tiny", "r=61.9"),
    ("TinyV9b",       40.5757, 0.6409,  21.03, "old", "tiny", "r=12.4"),
    ("PixB",          18.9710, 0.3143, 155.9,  "old", "pix",  "家族外"),
    ("TinyV21",       68.6681, 0.6306,   4.19, "new", "ref",  "学生单独"),
    ("TinyRefV3",     70.1021, 0.7952,   6.68, "new", "ref",  "thin1"),
    ("TinyRefV4",     69.4540, 0.8046,   7.00, "new", "ref",  "thin2"),
    ("TinyRefV4us0",  68.6300, 0.7846,   6.60, "new", "ref",  "去 unsharp（clipiqa 由实测 delta 推）"),
]
BASE, C0, P = 285.6, 2.0, 0.19
factor = lambda t, p=P, c0=C0: (BASE/(t+c0)/0.78)**p
IT = lambda s, t, p=P, c0=C0: s/factor(t, p, c0)

print("== 逐点反解 IT（p=0.19, c0=2.0）==")
for n, s, c, t, era, fam, note in D:
    print("   %-14s %-4s %-5s score=%8.4f c=%.4f t=%8.2f factor=%.3f IT=%7.2f" % (n, era, fam, s, c, t, factor(t), IT(s, t)))

def fit_loglog(pts):
    xs = [math.log(c-0.2752) for _, c, _ in pts]; ys = [math.log(v) for _, _, v in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    b = sxy/sxx; lA = my - b*mx
    ss_tot = sum((y-my)**2 for y in ys); ss_res = sum((y-(lA+b*x))**2 for x, y in zip(xs, ys))
    return math.exp(lA), b, 1-ss_res/ss_tot

print()
print("== 各子集 IT(clipiqa) 幂律拟合（检验单曲线是否成立）==")
subsets = {
  "新时代 refiner 线 (n=4)": [(n, c, IT(s, t)) for n, s, c, t, e, f, _ in D if e == "new"],
  "旧时代 tiny 家族 (n=6)": [(n, c, IT(s, t)) for n, s, c, t, e, f, _ in D if e == "old" and f == "tiny"],
  "旧时代 m24 家族 (n=7)": [(n, c, IT(s, t)) for n, s, c, t, e, f, _ in D if e == "old" and f == "m24"],
  "旧时代 tiny+m24 (n=13)": [(n, c, IT(s, t)) for n, s, c, t, e, f, _ in D if e == "old" and f in ("tiny","m24")],
  "全部 (n=18)": [(n, c, IT(s, t)) for n, s, c, t, e, f, _ in D],
}
for k, pts in subsets.items():
    A, b, r2 = fit_loglog(pts)
    print("   %-26s b=%6.3f  A=%6.1f  R2=%5.3f  (n=%d)" % (k, b, A, r2, len(pts)))

print()
print("== 速度指数 p 独立标定（同时代同图，仅换测速件）==")
pairs = [("M24_v3", 366.2, 43.6951), ("V7ganrun", 4.62, 94.7249), ("V4grun", 1.60, 105.6401), ("Slow8", 2929.6, 28.7538)]
ps = []
for (n1, t1, s1), (n2, t2, s2) in itertools.combinations(pairs, 2):
    pv = math.log(s1/s2)/math.log((t2+C0)/(t1+C0)); ps.append(pv)
    print("   %-10s vs %-10s : p = %.3f" % (n1, n2, pv))
print("   ⇒ p 的 6 个两两估计: min %.3f max %.3f mean %.3f（跨度 %.3f）" % (min(ps), max(ps), statistics.mean(ps), max(ps)-min(ps)))

print()
print("== 乘法 vs 加法结构（仅新时代 4 点，速度跨度 1.7x）==")
for n, s, c, t, e, f, _ in D:
    if e != "new": continue
    S_add = 2.63*math.log2(BASE/(t+C0)/0.78)
    print("   %-14s clipiqa=%.4f | 乘法隐含 IT=%6.2f | 加法隐含 IT=%6.2f" % (n, c, IT(s, t), s - S_add))
print()
print("== 饱和性检验：不同子集给出的斜率差异有多大 ==")
A1,b1,r21 = fit_loglog([(n,c,IT(s,t)) for n,s,c,t,e,f,_ in D if e=="new"])
A2,b2,r22 = fit_loglog([(n,c,IT(s,t)) for n,s,c,t,e,f,_ in D if e=="old" and f in ("tiny","m24")])
print("   新时代 refiner 线 b=%.3f（平缓） vs 旧时代跨家族 b=%.3f（陡） ⇒ 斜率估计**不稳定**" % (b1, b2))
print("   若用 b=%.3f：+0.05 clipiqa 值 %.1f%% IT；若用 b=%.3f：值 %.1f%% IT" %
      (b1, 100*((0.8546/0.8046)**b1-1), b2, 100*((0.8546/0.8046)**b2-1)))
print("E1_DONE")
