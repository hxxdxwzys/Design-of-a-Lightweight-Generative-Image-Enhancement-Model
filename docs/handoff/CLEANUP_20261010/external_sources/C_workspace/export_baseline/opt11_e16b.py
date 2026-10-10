r"""E16b: 置换检验完整打印——记录映射在 120 种分配中的名次（clipiqa / musiq 两个判据）。"""
import itertools, json, math, statistics
from pathlib import Path
BASE, C0, P = 285.6, 2.0, 0.19
fac = lambda t: (BASE / (t + C0) / 0.78) ** P
cands = [("TinyV21", 4.19, 0.6306, 48.6138), ("TinyRefV3", 6.68, 0.7952, 55.8474),
         ("TinyRefV4", 7.00, 0.8042, 57.6707), ("TinyRefV5", 6.66, 0.7775, 57.1235),
         ("RefV4us0", 6.60, 0.7918, 53.4782)]
sc = {"TinyV21": 68.6681, "TinyRefV3": 70.1021, "TinyRefV4": 69.4540, "TinyRefV5": 66.2813, "RefV4us0": 68.6300}
records = ["TinyV21", "TinyRefV3", "TinyRefV4", "TinyRefV5", "RefV4us0"]   # 记录映射：按时间序分配
def fit_power(xs, ys):
    lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    sxx = sum((x-mx)**2 for x in lx); sxy = sum((x-mx)*(y-my) for x, y in zip(lx, ly))
    b = sxy/sxx; lA = my - b*mx
    ss = sum((y-my)**2 for y in ly); sr = sum((y-(lA+b*x))**2 for x, y in zip(lx, ly))
    return math.exp(lA), b, (1-sr/ss if ss > 0 else float("nan"))
for key, name in ((2, "clipiqa"), (3, "musiq")):
    rows = []
    for perm in itertools.permutations(records):
        ITs = [sc[perm[i]]/fac(cands[i][1]) for i in range(5)]
        xs = [cands[i][key] for i in range(5)]
        A, b, r2 = fit_power(xs, ITs)
        sr = sum((math.log(y) - math.log(A*x**b))**2 for x, y in zip(xs, ITs))
        rows.append((sr, r2, perm, [round(x, 2) for x in ITs]))
    rows.sort()
    rec = [r for r in rows if r[2] == tuple(records)][0]
    print("== 判据 %s ==" % name)
    for i, (sr, r2, perm, ITs) in enumerate(rows[:5]):
        print("   #%d R2=%.3f 残差=%.5f  分配=%s" % (i+1, r2, sr, " ".join("%s<-%s" % (cands[j][0], perm[j]) for j in range(5))))
    print("   记录映射名次 = #%d / 120 （R2=%.3f 残差=%.5f）" % (rows.index(rec)+1, rec[1], rec[0]))
    print("   记录映射下的隐含 IT =", rec[3])
    print()
Path(r"D:\models\opt11\e16b_perm.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
print("E16B_DONE")
