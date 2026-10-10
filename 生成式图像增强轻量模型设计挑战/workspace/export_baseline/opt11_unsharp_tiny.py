# -*- coding: utf-8 -*-
"""在 v18 的 test30 出图上测轻度 unsharp（与 OPT-10 runner 内实现同式），产出可量目录。"""
import os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

SRC = Path(r"D:\models\opt11\eval_v18_s25000")
OUT = Path(r"D:\models\opt11")
dev = "cuda"

def unsharp(t, sigma=1.2, percent=60.0, thr_u8=3.0):
    ksize = int(round(sigma * 3)) * 2 + 1
    x = torch.arange(ksize, dtype=torch.float64) - (ksize - 1) / 2
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma)); k = (k / k.sum()).to(device=t.device, dtype=t.dtype)
    p = ksize // 2; c = int(t.shape[1])
    y = F.pad(t, (p, p, 0, 0), mode='reflect')
    y = F.conv2d(y, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    y = F.pad(y, (0, 0, p, p), mode='reflect')
    blur = F.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    diff = t - blur
    mask = (diff.abs() > thr_u8 / 255.0 * 2.0).to(t.dtype)
    out = (t + (percent / 100.0) * diff * mask).clamp(-1.0, 1.0)
    u8 = torch.round((out + 1.0) * 127.5)
    return (u8 / 127.5) - 1.0

files = sorted(SRC.glob("case*.jpg"))
print("源图", len(files), flush=True)
for pct in (60.0, 100.0):
    d = OUT / ("eval_v18_s25000_us%d" % int(pct)); d.mkdir(parents=True, exist_ok=True)
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        t = torch.from_numpy(a).permute(2, 0, 1)[None].to(dev).float().div_(127.5).sub_(1)
        with torch.no_grad():
            y = unsharp(t, percent=pct)
        o = ((y[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(o).save(d / f.name, quality=98, subsampling=0)
    print("写出", d, flush=True)
print("UNSHARP_TEST_DONE", flush=True)
