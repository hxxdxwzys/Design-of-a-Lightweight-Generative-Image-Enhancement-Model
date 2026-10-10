r"""E18: 用全部 14 个官方点估计"我们工况点附近"的质量敏感度 b，并据此评估 E17 候选。
疑问：b 取决于模型族/量程——跨族律 b≈0.97，族内(新时代5点) b≈0.30。到底该用哪个？
做法：比较三类模型在全 14 点上的表现，并给出各点的局部对数斜率 dlnIT/dlnc：
  (1) 纯幂律 IT=A·c^b（全 14 点）
  (2) 幂律 + 饱和 Hill：IT = ITmax·x^g/(K+x^g), x=c-0.28
  (3) 指数饱和 IT = ITmax·(1-exp(-k·x))
再用每个模型的**局部斜率**把 E17 的时延收益换算成总分比。
"""
import json, math, statistics
from pathlib import Path
import numpy as np

BASE, C0, P = 285.6, 2.0, 0.19
fac = lambda t: (BASE/(t+C0)/0.78)**P
# 14 个官方点：(tag, 官方分, 时延ms, clipiqa, 时代)
D = [
 ("TinyV1",27.3521,44.83,0.3367,"old"), ("TinyV2",29.4225,12.65,0.3532,"old"),
 ("TinyV2g",33.2872,3.30,0.3532,"old"), ("TinyV5d",33.0382,3.30,0.3897,"old"),
 ("TinyV7gan",41.7346,2.61,0.5059,"old"), ("TinyV9b",40.5757,21.03,0.6409,"old"),
 ("PixB",18.9710,155.90,0.3143,"old"), ("M24img",43.6951,366.20,0.9023,"old"),
 ("OPT10img",45.6141,324.00,0.9123,"old"),
 ("TinyV21",68.6681,4.19,0.6306,"new"), ("TinyRefV3",70.1021,6.68,0.7952,"new"),
 ("TinyRefV4",69.4540,7.00,0.8042,"new"), ("TinyRefV5",66.2813,6.66,0.7775,"new"),
 ("RefV4us0",68.6300,6.60,0.7918,"new"),
]
pts = [(t, s/fac(tt), c, era) for t,s,tt,c,era in D]
C = np.array([p[2] for p in pts]); Y = np.array([math.log(p[1]) for p in pts])

def r2(pred_log):
    ss = ((Y-Y.mean())**2).sum(); sr = ((Y-pred_log)**2).sum(); return 1-sr/ss

print("== 模型 (1) 纯幂律（全 14 点）==")
X = np.column_stack([np.ones(len(C)), np.log(C)])
coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
p1 = X@coef; b1 = coef[1]; A1 = math.exp(coef[0])
print("   A=%.3f b=%.3f R2=%.3f  (局部斜率恒为 %.3f)" % (A1, b1, r2(p1), b1))

print()
print("== 模型 (2) 幂律 + 静态截距（区分族）==")
era = np.array([1.0 if p[3]=="new" else 0.0 for p in pts])
X2 = np.column_stack([np.ones(len(C)), np.log(C), era])
coef2, *_ = np.linalg.lstsq(X2, Y, rcond=None)
print("   A=%.3f b=%.3f 新时代截距=%.3f (×%.2f) R2=%.3f" % (math.exp(coef2[0]), coef2[1], coef2[2], math.exp(coef2[2]), r2(X2@coef2)))

print()
print("== 模型 (3) Hill 饱和（全 14 点，x=c-0.28）==")
x = C-0.28
best = None
for g in np.arange(0.2, 4.0, 0.02):
    for lK in np.arange(-4.0, 3.0, 0.05):
        K = math.exp(lK)
        z = np.log(x**g/(K+x**g))
        Xh = np.column_stack([np.ones(len(x)), z])
        coefh, *_ = np.linalg.lstsq(Xh, Y, rcond=None)
        rr = 1-((Y-Xh@coefh)**2).sum()/((Y-Y.mean())**2).sum()
        if best is None or rr > best[0]: best = (rr, g, K, math.exp(coefh[0]))
rr, g, K, ITmax = best
print("   ITmax=%.2f g=%.3f K=%.4f R2=%.3f" % (ITmax, g, K, rr))
def hill_local_slope(c):
    xx = c-0.28
    return g*K/(K+xx**g)     # d ln IT / d ln x  (x 与 c 的对数斜率近似: ×c/x)
def hill_IT(c):
    xx = c-0.28; return ITmax*xx**g/(K+xx**g)

print()
print("== E17 候选评估（时延取同会话 median）==")
CAND = [("V4(现行最佳)", 6.61, 0.8042, 1050), ("L256", 5.11, 0.6369, 277),
        ("L192", 3.04, 0.5370, 162), ("L128", 2.06, 0.4133, 101), ("V21(裸学生)", 5.12, 0.6306, 164)]
print("%-14s %7s %8s | %-24s | %-24s | %-24s" % ("变体","ms","clipiqa","幂律(b=%.2f)相对V4" % b1, "Hill局部斜率", "族内b=0.34(新时代)"))
rows = []
for name, t, c, lap in CAND:
    f = fac(t)
    def ratio(ITc):
        return ITc*f/(A1*CAND[0][2]**b1*fac(CAND[0][1]))
    ip = A1*c**b1; ih = hill_IT(c)
    r_pow = (ip*f)/(A1*CAND[0][2]**b1*fac(CAND[0][1]))
    r_hill = (ih*f)/(hill_IT(CAND[0][2])*fac(CAND[0][1]))
    xr = c/CAND[0][2]
    r_new = (xr**0.34)*f/fac(CAND[0][1])
    sl = hill_local_slope(c)*c/(c-0.28)
    rows.append((name, t, c, r_pow, r_hill, r_new, sl))
    print("%-14s %7.2f %8.4f | %+7.1f%% (IT%+6.1f%%) | 斜率%5.2f  %+7.1f%%   | %+7.1f%%" % (
        name, t, c, 100*(r_pow-1), 100*(ip/(A1*CAND[0][2]**b1)-1), sl, 100*(r_hill-1), 100*(r_new-1)))
print()
print("== 结论区间：三个模型下 L192/V4 与 L128/V4 的总分比 ==")
for nm, t, c, rp, rh, rn, sl in rows[1:]:
    print("   %-6s : 幂律%+.1f%%  Hill%+.1f%%  族内b0.34%+.1f%%  ⇒ 区间 [%+.1f%%, %+.1f%%]" % (
        nm, 100*(rp-1), 100*(rh-1), 100*(rn-1), 100*(min(rp,rh,rn)-1), 100*(max(rp,rh,rn)-1)))
Path(r"D:\models\opt11\e18_sensitivity.json").write_text(json.dumps({
  "power_law": {"A": A1, "b": b1, "R2": r2(p1)},
  "two_level": {"A": math.exp(coef2[0]), "b": coef2[1], "era_shift": math.exp(coef2[2]), "R2": r2(X2@coef2)},
  "hill": {"ITmax": ITmax, "g": g, "K": K, "R2": rr},
  "candidates": [{"name": r[0], "ms": r[1], "clipiqa": r[2], "ratio_power": round(r[3], 4), "ratio_hill": round(r[4], 4), "ratio_newera_b034": round(r[5], 4), "local_slope": round(r[6], 3)} for r in rows]},
  indent=1, ensure_ascii=False), encoding="utf-8")
print("E18_DONE")
