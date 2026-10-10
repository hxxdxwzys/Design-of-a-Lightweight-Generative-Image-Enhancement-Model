r"""E15: E8 vs V4 的多协议配对显著性检验（决定"+1.03 musiq 是否真实"）。
用法: python opt11_e15_pairs.py --a DIR_V4 --b DIR_E8 [--n 30]
输出: D:\models\opt11\e15_pairs.json
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"): os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="A 目录(如 V4)")
ap.add_argument("--b", required=True, help="B 目录(如 E8)")
ap.add_argument("--n", type=int, default=30)
a = ap.parse_args()
dev = "cuda"
ms = {k: pyiqa.create_metric(k, device=dev) for k in ("clipiqa", "niqe", "musiq")}
CROPS = [256, 512, 768, 1024]
def S(metric, t):
    with torch.no_grad(): return float(ms[metric](t).item())
def tens(arr):
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)

names = [p.stem for p in sorted(Path(a.a).glob("*.jpg"))][:a.n]
data = {}
t0 = time.time()
for tag, d in (("A", a.a), ("B", a.b)):
    per = {("c%d" % c): [] for c in CROPS}; per["full"] = []; per["clipiqa4k"] = []; per["niqe4k"] = []
    for nm in names:
        p = Path(d) / ("%s.jpg" % nm)
        if not p.exists():
            print("MISS %s %s" % (tag, nm), flush=True); continue
        im = Image.open(p).convert("RGB"); arr = np.asarray(im); h, w = arr.shape[:2]
        for c in CROPS:
            if c > min(h, w): per["c%d" % c].append(None); continue
            y0, x0 = (h - c) // 2, (w - c) // 2
            per["c%d" % c].append(S("musiq", tens(arr[y0:y0+c, x0:x0+c])))
        s_ = 512.0 / max(h, w)
        im2 = im.resize((int(round(w*s_/8))*8, int(round(h*s_/8))*8), Image.BICUBIC)
        per["full"].append(S("musiq", tens(np.asarray(im2))))
        t4 = tens(arr)
        per["clipiqa4k"].append(S("clipiqa", t4)); per["niqe4k"].append(S("niqe", t4))
        del t4, im, im2
    data[tag] = per
    print("[%s] %s done %.0fs" % (tag, d, time.time()-t0), flush=True)

def binom_p(k, n):
    # 双尾符号检验 p 值
    from math import comb
    if n == 0: return None
    obs = comb(n, k)
    tot = sum(comb(n, i) for i in range(n+1))
    p = sum(comb(n, i) for i in range(n+1) if comb(n, i) <= obs) / tot
    return round(min(1.0, p), 4)

res = {"names": names, "a": a.a, "b": a.b}
print("\n%-10s %10s %10s %9s %6s %6s" % ("协议", "A均值", "B均值", "B-A", "B胜", "p"))
for k in list(data["A"].keys()):
    va, vb = data["A"][k], data["B"][k]
    pair = [(x, y) for x, y in zip(va, vb) if x is not None and y is not None]
    if not pair: continue
    ma = float(np.mean([x for x, _ in pair])); mb = float(np.mean([y for _, y in pair]))
    wins = sum(1 for x, y in pair if y > x)
    diff = [y - x for x, y in pair]
    res[k] = {"A_mean": round(ma, 4), "B_mean": round(mb, 4), "B_minus_A": round(mb-ma, 4),
              "B_wins": wins, "n": len(pair), "sign_p": binom_p(wins, len(pair)),
              "diff_mean": round(float(np.mean(diff)), 4), "diff_sd": round(float(np.std(diff, ddof=1)), 4),
              "paired_t": round(float(np.mean(diff)/(np.std(diff, ddof=1)/np.sqrt(len(diff)))), 2) if len(diff) > 2 else None}
    print("%-10s %10.3f %10.3f %9.3f %6s %6s" % (k, ma, mb, mb-ma, "%d/%d" % (wins, len(pair)), res[k]["sign_p"]), flush=True)
Path(r"D:\models\opt11\e15_pairs.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print("E15_DONE", flush=True)
