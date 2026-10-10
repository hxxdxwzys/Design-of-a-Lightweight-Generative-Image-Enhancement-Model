r"""E2b: 分时代拟合 + 尺度一致性检验：哪个代理能跨时代一致地解释官方 IT？"""
import json, math, statistics
import numpy as np
from pathlib import Path
P = Path(r"D:\models\opt11")
OFF = {   # tag: (官方分, t_ms, 时代)
 "TinyV1":(27.3521,44.83,"old"), "TinyV2":(29.4225,12.65,"old"), "TinyV2g":(33.2872,3.30,"old"),
 "TinyV5d":(33.0382,3.30,"old"), "TinyV7gan":(41.7346,2.61,"old"), "TinyV9b":(40.5757,21.03,"old"),
 "PixB":(18.9710,155.9,"old"), "M24img":(43.6951,366.2,"old"), "OPT10img":(45.6141,324.0,"old"),
 "NEW_TinyV21":(68.6681,4.19,"new"), "NEW_TinyRefV3":(70.1021,6.68,"new"), "NEW_TinyRefV4":(69.4540,7.00,"new"),
 "NEW_TinyRefV5":(66.2813,6.66,"new"), "NEW_TinyRefV4us0":(68.6300,6.60,"new"),
}
BASE, C0, p = 285.6, 2.0, 0.19
factor = lambda t: (BASE/(t+C0)/0.78)**p
rows = []
for tag,(s,t,era) in OFF.items():
    f = P/("proxy_%s.json"%tag)
    if not f.exists(): print("miss", tag); continue
    d = json.loads(f.read_text(encoding="utf-8"))
    rows.append({"tag":tag,"era":era,"IT":s/factor(t), **{k:d[k] for k in ("clipiqa","niqe","musiq","maniqa","lapvar")}})

def fit_power(xs, ys):
    lx=[math.log(x) for x in xs]; ly=[math.log(y) for y in ys]
    mx,my=statistics.mean(lx),statistics.mean(ly)
    sxx=sum((x-mx)**2 for x in lx); sxy=sum((x-mx)*(y-my) for x,y in zip(lx,ly))
    b=sxy/sxx; lA=my-b*mx
    ss=sum((y-my)**2 for y in ly); sr=sum((y-(lA+b*x))**2 for x,y in zip(lx,ly))
    return math.exp(lA), b, 1-sr/ss

print("== 分时代拟合（幂律 IT=A·x^b）==")
print("%-8s | %-34s | %-34s" % ("proxy","旧时代 (n=9)","新时代 (n=5)"))
for k in ("clipiqa","musiq","maniqa","lapvar","niqe"):
    o=[r for r in rows if r["era"]=="old"]; n=[r for r in rows if r["era"]=="new"]
    Ao,bo,Ro=fit_power([r[k] for r in o],[r["IT"] for r in o])
    An,bn,Rn=fit_power([r[k] for r in n],[r["IT"] for r in n])
    print("%-8s | A=%8.4f b=%6.3f R2=%5.3f      | A=%8.4f b=%6.3f R2=%5.3f" % (k,Ao,bo,Ro,An,bn,Rn))

print()
print("== 尺度一致性检验：用旧时代曲线预测新时代点，看隐含尺度 s=IT_new/IT_pred ==")
for k in ("clipiqa","musiq","maniqa","lapvar"):
    o=[r for r in rows if r["era"]=="old"]
    A,b,_=fit_power([r[k] for r in o],[r["IT"] for r in o])
    ss=[]
    for r in rows:
        if r["era"]!="new": continue
        pred=A*(r[k]**b); ss.append(r["IT"]/pred)
    print("   %-8s 隐含 s: mean=%.3f  sd=%.3f  min=%.3f max=%.3f  (变异系数 %.1f%%)" %
          (k, statistics.mean(ss), statistics.pstdev(ss), min(ss), max(ss), 100*statistics.pstdev(ss)/statistics.mean(ss)))

print()
print("== 若允许一个统一尺度 s，各代理的总体拟合优度（14 点）==")
for k in ("clipiqa","musiq","maniqa","lapvar","niqe"):
    o=[r for r in rows if r["era"]=="old"]; A,b,_=fit_power([r[k] for r in o],[r["IT"] for r in o])
    # 最优 s（最小化对数残差平方）
    lr=[math.log(r["IT"]/(A*r[k]**b)) for r in rows]
    s=math.exp(statistics.mean(lr))
    ss=sum((x-math.log(s))**2 for x in lr); tot=sum((math.log(r["IT"])-statistics.mean([math.log(q["IT"]) for q in rows]))**2 for r in rows)
    print("   %-8s 最优 s=%.3f  对数残差平方=%.4f  (占总体 %.1f%%)" % (k, s, ss, 100*ss/tot))
print("E2B_DONE")
