# -*- coding: utf-8 -*-
"""配对显著性检验：两组渲染目录逐图比较（符号检验 + 均值差 + 胜负计数）。
用法: python paired.py <dirA> <dirB> [--ref-dir / --metrics clipiqa,niqe]
逐图算 clipiqa / niqe（4K 全图），做配对符号检验。
"""
import argparse, os, sys, json
os.environ["HF_HUB_OFFLINE"] = "1"
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image
from scipy import stats as st

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser()
ap.add_argument("A"); ap.add_argument("B")
ap.add_argument("--metrics", default="clipiqa,niqe")
ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
dev = "cuda"
want = [m.strip() for m in a.metrics.split(",")]
ms = {k: pyiqa.create_metric(k, device=dev) for k in want}


def measure(d):
    files = sorted(os.listdir(d), key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))
    if a.n:
        files = files[:a.n]
    out = {k: [] for k in want}
    for f in files:
        arr = np.asarray(Image.open(os.path.join(d, f)).convert("RGB"))
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
        for k, m in ms.items():
            with torch.no_grad():
                out[k].append(float(m(t).item()))
        del t
    return files, out


fa, va = measure(a.A)
fb, vb = measure(a.B)
print("[paired] n=%d  A=%s  B=%s" % (len(fa), os.path.basename(a.A), os.path.basename(a.B)), flush=True)
print("%-9s %10s %10s %10s %8s %8s %10s" % ("metric", "meanA", "meanB", "delta", "A胜", "B胜", "p(符号)"))
for k in want:
    x = np.array(va[k]); y = np.array(vb[k])
    # 方向：clipiqa 越高越好；niqe 越低越好
    lower_better = (k == "niqe")
    d = (y - x) if not lower_better else (x - y)   # d>0 表示 B 更好
    nz = d[d != 0]
    winsB = int((nz > 0).sum()); winsA = int((nz < 0).sum())
    p = st.binomtest(winsB, len(nz), 0.5).pvalue if len(nz) else 1.0
    print("%-9s %10.4f %10.4f %+10.4f %8d %8d %10.4f" % (k, x.mean(), y.mean(), y.mean() - x.mean(), winsA, winsB, p))
print("PAIRED_DONE")
