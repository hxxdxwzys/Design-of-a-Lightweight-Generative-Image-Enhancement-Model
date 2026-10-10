r"""E22: 全局候选合并排序（审计收官产物）。
把每个变体的 时延 归一到"V4=6.47 ms"同会话尺度（用各次 lat3 会话内的 V4 锚点比例），
再在 b∈{0.34(新时代族内), 0.97(跨族)} 两个质量敏感度下计算相对 V4 的总分比。
排序按 min(两个净值)（保守）。
"""
import json, math
from pathlib import Path
P = Path(r"D:\models\opt11")
BASE, C0, PW = 285.6, 2.0, 0.19
fac = lambda t: (BASE/(t+C0)/0.78)**PW
V4_LAT = 6.47          # 当前基准（最近一次 lat3 的 V4 median）
V4_CLI = 0.8042
# (标签, proxy文件, 本会话 V4 锚点 median（同会话）, 该变体 median, 备注)
ROWS = [
 ("V4 (thin2 16ch+us60) 已提交", "NEW_TinyRefV4", 6.47, 6.47, "基准"),
 ("V21 (学生裸) 已提交",        "NEW_TinyV21",   6.61, 5.12, "E17 会话"),
 ("V3 (thin1) 已提交",          "NEW_TinyRefV3", 6.33, 6.68, "旧会话(近似)"),
 ("V5 (w32) 已提交",            "NEW_TinyRefV5", 6.33, 6.66, "旧会话(近似)"),
 ("V4us0 (关 unsharp) 已提交",   "NEW_TinyRefV4us0", 6.33, 6.60, "旧会话(近似)"),
 ("E8 (VOSR 目标 32ch)",        "E8_vosr1",      6.33, 6.72, "E8 会话"),
 ("L256 (无 refiner+us60)",     "E17_L256",      6.61, 5.11, "E17 会话"),
 ("L192",                        "E17_L192",      6.61, 3.04, "E17 会话"),
 ("L128",                        "E17_L128",      6.61, 2.06, "E17 会话"),
 ("R04ch",                       "E19_R04ch",     6.47, 5.86, "E19 会话"),
 ("R08ch",                       "E19_R08ch",     6.47, 6.22, "E19 会话"),
 ("R16ch (新训)",                "E19_R16ch",     6.47, 6.59, "E19 会话"),
 ("r256c (256域 refiner)",       "E5_r256c_10k",  6.33, 6.68, "旧会话(近似)"),
 ("r256d (256域 48ch)",          "E6_r256d_48ch", 6.33, 6.14, "旧会话(近似)"),
 ("w32@10k",                     "E5_w32_10k",    6.33, 6.60, "旧会话(近似)"),
 ("E12 (学生512域)",             "E12_stu512",    6.33, 8.50, "旧会话(近似)"),
 ("E11 (VOSR 出图+refiner)",     "E11_v4ref_on_vosr", 6.33, 8.50, "旧会话(近似)"),
]
out = []
for name, tag, v4lst, lat, note in ROWS:
    f = P / ("proxy_%s.json" % tag)
    if not f.exists(): print("MISS", tag); continue
    d = json.loads(f.read_text(encoding="utf-8"))
    lat_n = lat * V4_LAT / v4lst          # 归一到 V4=6.47 的尺度
    c = d["clipiqa"]
    fr = fac(lat_n) / fac(V4_LAT)
    for b in (0.34, 0.97):
        itr = (c / V4_CLI) ** b
        out.append({"name": name, "b": b, "net": itr * fr, "it": itr, "fac": fr,
                    "cli": c, "musiq": d["musiq"], "niqe": d["niqe"], "lap": d["lapvar"], "ms": round(lat_n, 2)})
best = {}
for r in out:
    k = r["name"]
    best.setdefault(k, {"b034": None, "b097": None, "cli": r["cli"], "musiq": r["musiq"], "niqe": r["niqe"], "lap": r["lap"], "ms": r["ms"]})
    best[k]["b034" if r["b"] == 0.34 else "b097"] = r["net"]
rows = sorted(best.items(), key=lambda kv: -min(kv[1]["b034"], kv[1]["b097"]))
print("%-28s %6s %7s %7s %6s | %8s %8s | %s" % ("变体", "ms", "clipiqa", "musiq", "niqe", "净值b.34", "净值b.97", "min"))
for k, v in rows:
    print("%-28s %6.2f %7.4f %7.2f %6.3f | %+7.1f%% %+7.1f%% | %+7.1f%%" % (
        k, v["ms"], v["cli"], v["musiq"], v["niqe"], 100*(v["b034"]-1), 100*(v["b097"]-1), 100*(min(v["b034"], v["b097"])-1)))
Path(r"D:\models\opt11\e22_ranking.json").write_text(json.dumps(
    [{"name": k, "ms": v["ms"], "clipiqa": v["cli"], "musiq": v["musiq"], "niqe": v["niqe"],
      "net_b034": round(v["b034"], 4), "net_b097": round(v["b097"], 4)} for k, v in rows], indent=1, ensure_ascii=False), encoding="utf-8")
print("E22_DONE")
