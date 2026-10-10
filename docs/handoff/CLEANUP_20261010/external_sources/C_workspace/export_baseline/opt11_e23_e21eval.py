r"""E23: E21（可微 CLIP-IQA 奖励）评估 —— 与 V4 对比（多尺度 + 时延归一）。"""
import json
from pathlib import Path
P = Path(r"D:\models\opt11")
BASE, C0, PW = 285.6, 2.0, 0.19
fac = lambda t: (BASE/(t+C0)/0.78)**PW
lat = json.loads((P/"lat3.json").read_text(encoding="utf-8"))["summary"] if (P/"lat3.json").exists() else {}
def med(sub):
    for k, v in lat.items():
        if sub in k: return v["median"]
    return None
V4 = med("TinyRefV4") or 6.47
def cropclip(tag):
    f = P/("cropclip_%s.json" % tag)
    return json.loads(f.read_text(encoding="utf-8"))["clipiqa_crop_mean"] if f.exists() else None
rows = []
for name, ptag, ltag, ctag in (("V4 (基准)", "NEW_TinyRefV4", "TinyRefV4", None),
                               ("clip1 (从头 w5)", "E21_clip1", "pkg_clip1", "E21_clip1"),
                               ("clip2 (微调 w20)", "E21_clip2", "pkg_clip2", "E21_clip2"),
                               ("E8 (VOSR 目标)", "E8_vosr1", None, None),
                               ("L256 (无 refiner)", "E17_L256", "L256", None)):
    f = P/("proxy_%s.json" % ptag)
    if not f.exists(): print("MISS", ptag); continue
    d = json.loads(f.read_text(encoding="utf-8"))
    t = med(ltag) if ltag else None
    rows.append({"name": name, "cli": d["clipiqa"], "musiq": d["musiq"], "niqe": d["niqe"], "maniqa": d["maniqa"],
                 "lap": d["lapvar"], "ms": t, "croclip": cropclip(ctag) if ctag else None})
base = rows[0]
print("%-20s %7s %8s %8s %7s %7s %8s | %8s %8s %8s" % ("变体", "ms", "clipiqa", "crop-cli", "musiq", "niqe", "lapvar", "IT比b.34", "IT比b.97", "净值b.34"))
for r in rows:
    if r["ms"] is None: r["ms"] = float("nan")
    if base["ms"] and r["ms"] == r["ms"]:
        fr = fac(r["ms"])/fac(base["ms"])
    else:
        fr = float("nan")
    i34 = (r["cli"]/base["cli"])**0.34; i97 = (r["cli"]/base["cli"])**0.97
    net = i34*fr if fr == fr else float("nan")
    print("%-20s %7.2f %8.4f %8s %7.2f %7.3f %8.1f | %+7.1f%% %+7.1f%% %+8.1f%%" % (
        r["name"], r["ms"], r["cli"], ("%.4f" % r["croclip"]) if r["croclip"] else "-", r["musiq"], r["niqe"], r["lap"],
        100*(i34-1), 100*(i97-1), 100*(net-1) if net == net else float("nan")))
Path(P/"e23_e21_eval.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
print("E23_DONE")
