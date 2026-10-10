r"""E5 收口分析：用官方 14 点拟合综合代理 IT=A·clipiqa^b1·musiq^b2（含/不含 era 尺度），再对全部变体排名。"""
import json, math, statistics
import numpy as np
from pathlib import Path
P = Path(r"D:\models\opt11")
BASE, C0, p = 285.6, 2.0, 0.19
factor = lambda t: (BASE/(t+C0)/0.78)**p
OFF = {
 "TinyV1":(27.3521,44.83), "TinyV2":(29.4225,12.65), "TinyV2g":(33.2872,3.30), "TinyV5d":(33.0382,3.30),
 "TinyV7gan":(41.7346,2.61), "TinyV9b":(40.5757,21.03), "PixB":(18.9710,155.9),
 "M24img":(43.6951,366.2), "OPT10img":(45.6141,324.0),
 "NEW_TinyV21":(68.6681,4.19), "NEW_TinyRefV3":(70.1021,6.68), "NEW_TinyRefV4":(69.4540,7.00),
 "NEW_TinyRefV5":(66.2813,6.66), "NEW_TinyRefV4us0":(68.6300,6.60),
}
rows=[]
for tag,(s,t) in OFF.items():
    f=P/("proxy_%s.json"%tag)
    if not f.exists(): print("miss",tag); continue
    d=json.loads(f.read_text(encoding="utf-8"))
    rows.append({"tag":tag,"IT":s/factor(t),"c":d["clipiqa"],"m":d["musiq"],"mn":d["maniqa"],"lv":d["lapvar"]})
X=np.array([[1.0, math.log(r["c"]), math.log(r["m"])] for r in rows]); Y=np.array([math.log(r["IT"]) for r in rows])
coef,*_=np.linalg.lstsq(X,Y,rcond=None); pred=X@coef
r2=1-float(((Y-pred)**2).sum())/float(((Y-Y.mean())**2).sum())
print("== 综合代理拟合（14 点）IT = A·clipiqa^b1·musiq^b2 ==")
print("   A=%.4f b1=%.3f b2=%.3f   R2=%.4f   (对比: musiq 单变量 6.1%% 残差, clipiqa 9.4%%)" % (math.exp(coef[0]), coef[1], coef[2], r2))
print("   逐点残差:")
for i,r in enumerate(rows):
    pr=math.exp(pred[i]); print("      %-16s IT=%6.2f 预测=%6.2f  %+5.1f%%" % (r["tag"], r["IT"], pr, 100*(pr/r["IT"]-1)))

def ITp(c, m):
    return math.exp(coef[0] + coef[1]*math.log(c) + coef[2]*math.log(m))
CAND = [
 ("TinyV21 学生单独(已交)",       0.6306, 48.61, 164.2,  4.19),
 ("V3 thin1 16ch/2bl (已交)",    0.7946, 55.85, 796.8,  6.68),
 ("V4 thin2 16ch/2bl (已交)",    0.8042, 57.67, 1050.4, 7.00),
 ("V5 w32@6k (已交)",            0.7815, 57.12, 1039.6, 6.66),
 ("V4us0 (已交)",                0.7918, 53.48, 510.4,  6.60),
 ("w32@10k（此前被 clipiqa 否决）", 0.7775, 57.21, 978.2,  6.66),
 ("gan1 48ch（被时延否决）",       0.7594, 58.02, 1011.4, 17.40),
 ("v1 48ch L1+LPIPS（被 clipiqa 否决）", 0.7204, 56.29, 374.4, 17.40),
 ("r256c 32ch 256域（被 clipiqa 否决）", 0.6790, 55.88, 1299.3, 5.69),
]
print()
print("== 全变体排名（按综合代理预测分 = IT_pred × factor）==")
print("%-34s %7s %7s %8s %8s %8s %8s" % ("变体","clipiqa","musiq","lapvar","t(ms)","factor","预测分"))
res=[]
for name,c,m,lv,t in CAND:
    it=ITp(c,m); sc=it*factor(t); res.append((sc,name,c,m,lv,t,factor(t),it))
for sc,name,c,m,lv,t,f,it in sorted(res, reverse=True):
    print("%-34s %7.4f %7.2f %8.0f %8.2f %8.3f %8.2f   (IT_pred=%.2f)" % (name,c,m,lv,t,f,sc,it))
print()
print("== 已提交件：预测 vs 官方 ==")
for tag,(s,t) in OFF.items():
    if not tag.startswith("NEW"): continue
    f=P/("proxy_%s.json"%tag); d=json.loads(f.read_text(encoding="utf-8"))
    pr=ITp(d["clipiqa"], d["musiq"])*factor(t)
    print("   %-16s 官方 %7.4f | 综合代理预测 %7.4f (%+5.1f%%)" % (tag, s, pr, 100*(pr/s-1)))
print("E5_DONE")
