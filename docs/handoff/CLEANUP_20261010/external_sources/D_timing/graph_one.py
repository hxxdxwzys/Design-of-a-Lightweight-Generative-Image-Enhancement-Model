# -*- coding: utf-8 -*-
"""单项 CUDA graph 捕获测试（每个进程只测一个模型，避免上下文污染）。
用法: python graph_one.py <modeldir> <label> [--eager-ref <other_modeldir>]
"""
import argparse, importlib.util, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser()
ap.add_argument("md"); ap.add_argument("label")
ap.add_argument("--eager-ref", default="")
ap.add_argument("--dump", default="")
a = ap.parse_args()
dev = "cuda"
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
img = np.asarray(Image.open(TEST / "case3.jpg").convert("RGB"), dtype=np.uint8)[:512, :512]
t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div_(127.5).sub_(1.0)

# 先只做 eager（不碰 graph）
m = torch.jit.load(os.path.join(a.md, "your_model.pt"), map_location=dev); m.eval()
x = t.to(dev).half()
with torch.no_grad():
    y = m(x)
print("[%s] eager OK out=%s range=[%.3f,%.3f]" % (a.label, tuple(y.shape), float(y.min()), float(y.max())), flush=True)

if a.dump:
    np.save(a.dump, y.float().cpu().numpy())
    print("[%s] dumped -> %s" % (a.label, a.dump), flush=True)

# 再尝试手动捕获
si = torch.zeros(1, 3, 512, 512, device=dev, dtype=torch.float16)
try:
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            m(si)
    torch.cuda.current_stream().wait_stream(s)
    print("[%s] warmup OK" % a.label, flush=True)
except Exception as e:
    print("[%s] warmup FAILED: %s" % (a.label, str(e)[:250]), flush=True); sys.exit(0)

try:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        so = m(si)
    print("[%s] CAPTURE OK" % a.label, flush=True)
except Exception as e:
    print("[%s] CAPTURE FAILED: %s" % (a.label, str(e)[:400]), flush=True)
print("GRAPH_ONE_DONE", a.label, flush=True)
