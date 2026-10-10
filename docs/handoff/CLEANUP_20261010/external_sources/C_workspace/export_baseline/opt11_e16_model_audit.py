r"""E16: 分数模型外推能力审计 + 官方分↔包映射的置换稳健性（纯 CPU）。
问题：会话内"修正模型"(IT=A·clipiqa^b·musiq^c) 在旧时代宽量程(IT 15~45)上拟合优，
      但它能否解释新时代 5 个官方点(IT 31.6~35.6)？若不能，则该模型对新时代变体的排序无判别力。
"""
import itertools, json, math, statistics
from pathlib import Path
import numpy as np

BASE, C0, P = 285.6, 2.0, 0.19
fac = lambda t: (BASE / (t + C0) / 0.78) ** P

# 旧时代 9 点（含官方分与实测时延、代理读数）
OLD = [
    ("TinyV1",   27.3521, 44.83, 0.3367, 41.55),
    ("TinyV2",   29.4225, 12.65, 0.3532, 39.37),
    ("TinyV2g",  33.2872,  3.30, 0.3532, 39.37),
    ("TinyV5d",  33.0382,  3.30, 0.3897, 41.26),
    ("TinyV7gan",41.7346,  2.61, 0.5059, 43.68),
    ("TinyV9b",  40.5757, 21.03, 0.6409, 50.68),
    ("PixB",     18.9710,155.90, 0.3143, 40.35),
    ("M24img",   43.6951,366.20, 0.9023, 62.81),
    ("OPT10img", 45.6141,324.00, 0.9123, 65.86),
]
# 新时代 5 点：官方分、时延、clipiqa、musiq（记录映射；V5 为低置信）
NEW = [
    ("TinyV21",  68.6681, 4.19, 0.6306, 48.6138),
    ("TinyRefV3",70.1021, 6.68, 0.7952, 55.8474),
    ("TinyRefV4",69.4540, 7.00, 0.8042, 57.6707),
    ("TinyRefV5",66.2813, 6.66, 0.7775, 57.1235),
    ("RefV4us0", 68.6300, 6.60, 0.7918, 53.4782),
]
old = [{"tag":t, "IT":s/fac(tt), "c":c, "m":m} for t,s,tt,c,m in OLD]
new = [{"tag":t, "IT":s/fac(tt), "c":c, "m":m, "score":s, "t":tt} for t,s,tt,c,m in NEW]

def fit_power(xs, ys):
    lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    sxx = sum((x-mx)**2 for x in lx); sxy = sum((x-mx)*(y-my) for x, y in zip(lx, ly))
    b = sxy/sxx; lA = my - b*mx
    ss = sum((y-my)**2 for y in ly); sr = sum((y-(lA+b*x))**2 for x, y in zip(lx, ly))
    return math.exp(lA), b, (1-sr/ss if ss > 0 else float("nan"))

def fit2(xs, zs, ys):   # log IT = lA + b1 log x + b2 log z
    X = np.column_stack([np.ones(len(xs)), np.log(xs), np.log(zs)]); y = np.log(ys)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss = ((y - y.mean())**2).sum(); sr = ((y - pred)**2).sum()
    return math.exp(coef[0]), coef[1], coef[2], 1 - sr/ss

def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra*rb).sum()/math.sqrt((ra**2).sum()*(rb**2).sum()))

print("== 旧时代拟合（n=9）与对新时代 5 点的外推误差 ==")
models = {}
A, b, r2 = fit_power([r["c"] for r in old], [r["IT"] for r in old]); models["clipiqa"] = ("p", A, b, r2)
A, b, r2 = fit_power([r["m"] for r in old], [r["IT"] for r in old]); models["musiq"] = ("p", A, b, r2)
A2, b1, b2, r2 = fit2([r["c"] for r in old], [r["m"] for r in old], [r["IT"] for r in old]); models["c×m"] = ("2", A2, b1, b2, r2)
for k, v in models.items():
    print("  %-8s 旧 R2=%.3f  参数=%s" % (k, v[-1], [round(x, 4) for x in v[1:-1]]))
print()
print("%-10s %8s | %-28s | %-28s | %-28s" % ("新时代点", "官方IT", "clipiqa外推(误差%)", "musiq外推(误差%)", "2变量外推(误差%)"))
preds = {"clipiqa": [], "musiq": [], "c×m": []}
for r in new:
    _, A, b, _ = models["clipiqa"]; p1 = A*r["c"]**b
    _, A, b, _ = models["musiq"];   p2 = A*r["m"]**b
    _, A, b1, b2, _ = models["c×m"]; p3 = A*r["c"]**b1*r["m"]**b2
    preds["clipiqa"].append(p1); preds["musiq"].append(p2); preds["c×m"].append(p3)
    print("%-10s %8.2f | %8.2f (%+6.1f%%)        | %8.2f (%+6.1f%%)        | %8.2f (%+6.1f%%)" % (
        r["tag"], r["IT"], p1, 100*(p1/r["IT"]-1), p2, 100*(p2/r["IT"]-1), p3, 100*(p3/r["IT"]-1)))
off = [r["IT"] for r in new]
print()
print("== 模型在【新时代内部】的排序能力（官方 IT 排序 vs 预测排序）==")
for k in preds:
    print("   %-8s Spearman(pred, 官方IT) = %+.3f   平均绝对误差 %.1f%%" % (
        k, spear(preds[k], off), 100*statistics.mean([abs(x/y-1) for x, y in zip(preds[k], off)])))

print()
print("== 新时代自身拟合（n=5，仅说明「能不能解释」，不能证明）==")
for key, name in (("c", "clipiqa"), ("m", "musiq")):
    A, b, r2 = fit_power([r[key] for r in new], [r["IT"] for r in new])
    print("   %-8s A=%8.4f b=%6.3f R2=%5.3f" % (name, A, b, r2))

print()
print("== 映射置换稳健性：120 种官方分↔包 分配，用 clipiqa 幂律(新时代内拟合)的残差排序 ==")
cands = [(r["tag"], r["t"], r["c"]) for r in new]
scores = [r["score"] for r in new]
rec = tuple(r["tag"] for r in new)
rows = []
for perm in itertools.permutations(range(5)):
    ITs = [scores[perm[i]]/fac(cands[i][1]) for i in range(5)]
    cs = [cands[i][2] for i in range(5)]
    A, b, r2 = fit_power(cs, ITs)
    sr = sum((math.log(y) - math.log(A*x**b))**2 for x, y in zip(cs, ITs))
    rows.append((sr, r2, tuple(cands[i][0] for i in range(5)), tuple(cands[i][0] for i in range(5) if True)))
rows.sort()
for i, (sr, r2, _, _) in enumerate(rows[:6]):
    assign = {cands[j][0]: scores[it[j]] for j in range(5)} if False else None
    print("   #%d 残差=%.5f R2=%5.3f" % (i+1, sr, r2))
# 记录映射的位置
rec_sr = None
for sr, r2, _, _ in rows:
    pass
print("   （记录映射 V21/V3/V4/V5/V4us0 的残差与名次见 JSON）")
out = {"old_fit": {k: [round(x, 5) for x in v[1:]] for k, v in models.items()},
       "new_pred": {k: [round(x, 3) for x in preds[k]] for k in preds},
       "new_IT": [round(x, 3) for x in off],
       "new_spearman": {k: round(spear(preds[k], off), 3) for k in preds},
       "perm_sorted": [{"resid": round(sr, 6), "r2": round(r2, 4)} for sr, r2, _, _ in rows[:6]]}
Path(r"D:\models\opt11\e16_model_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("E16_DONE")
