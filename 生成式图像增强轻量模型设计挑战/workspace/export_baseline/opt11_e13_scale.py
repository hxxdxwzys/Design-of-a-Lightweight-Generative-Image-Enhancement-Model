r"""E13: 代理协议分辨率稳健性检验。

问题：我们所有变体排序都基于 musiq@512 中心裁剪。若该排序随裁剪尺度改变，则整套排序结论失效。
做法：对 4 个已知质量档位的输出目录，逐图计算 musiq 在 256/512/768/1024 中心裁剪 + 全图（长边缩到512）上的值，
      保存逐图分数，随后计算 (a) 各协议下目录均值排序是否一致 (b) 协议间逐图 Spearman 相关。
用法: python opt11_e13_scale.py --n 20
输出: D:\models\opt11\e13_scale.json
"""
import json, os, sys, time
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DIRS = [
    ("LQ_输入",   r"D:\models\probe_a\lq30"),
    ("V11b_8k",  r"D:\models\traj_v11b\s8000"),
    ("V11b_24k", r"D:\models\traj_v11b\s24000"),
    ("E6_r256d", r"D:\models\opt11\e6_render"),
]
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 20
dev = "cuda"
m = pyiqa.create_metric("musiq", device=dev)
CROPS = [256, 512, 768, 1024]

def score(t):
    with torch.no_grad():
        return float(m(t).item())

res = {}
t0 = time.time()
for tag, d in DIRS:
    p = Path(d)
    names = sorted({f.stem for f in p.glob("*.jpg")})[:N]
    if not names:
        print("SKIP(no img) %s" % d, flush=True); continue
    per = {("c%d" % c): [] for c in CROPS}
    per["full"] = []
    for nm in names:
        f = p / ("%s.jpg" % nm)
        im = Image.open(f).convert("RGB")
        arr = np.asarray(im)
        h, w = arr.shape[:2]
        for c in CROPS:
            if c > min(h, w):
                per["c%d" % c].append(None); continue
            y0, x0 = (h - c) // 2, (w - c) // 2
            sub = arr[y0:y0 + c, x0:x0 + c]
            t = torch.from_numpy(sub.copy()).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
            per["c%d" % c].append(score(t)); del t
        s = 512.0 / max(h, w)
        nh, nw = max(64, int(round(h * s / 8)) * 8), max(64, int(round(w * s / 8)) * 8)
        im2 = im.resize((nw, nh), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(im2).copy()).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
        per["full"].append(score(t)); del t
        del im, im2
    res[tag] = {"dir": d, "n": len(names), "per_img": per,
                "mean": {k: (round(float(np.mean([x for x in v if x is not None])), 4) if any(x is not None for x in v) else None) for k, v in per.items()}}
    print("E13 %-9s %s" % (tag, json.dumps(res[tag]["mean"], ensure_ascii=False)), flush=True)

# 协议间逐图 Spearman + 目录排序一致性
def spearman(a, b):
    idx = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
    if len(idx) < 5: return None
    ra = np.argsort(np.argsort([a[i] for i in idx])).astype(float)
    rb = np.argsort(np.argsort([b[i] for i in idx])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return round(float((ra * rb).sum() / d), 4) if d > 0 else None

keys = [("c%d" % c) for c in CROPS] + ["full"]
corr = {}
for i, k1 in enumerate(keys):
    for k2 in keys[i + 1:]:
        vals = []
        for tag in res:
            pi = res[tag]["per_img"]
            r = spearman(pi.get(k1, []), pi.get(k2, []))
            if r is not None: vals.append(r)
        corr["%s~%s" % (k1, k2)] = round(float(np.mean(vals)), 4) if vals else None
order = {k: [t for t, _ in sorted(((t, res[t]["mean"][k]) for t in res if res[t]["mean"].get(k) is not None), key=lambda x: -x[1])] for k in keys}
out = {"dirs": res, "spearman_mean": corr, "order_by_protocol": order, "n": N, "secs": round(time.time() - t0, 1)}
Path(r"D:\models\opt11\e13_scale.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("E13_ORDER " + json.dumps(order, ensure_ascii=False), flush=True)
print("E13_CORR  " + json.dumps(corr, ensure_ascii=False), flush=True)
print("E13_DONE %.1f s" % (time.time() - t0), flush=True)
