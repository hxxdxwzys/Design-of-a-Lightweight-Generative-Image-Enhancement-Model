# -*- coding: utf-8 -*-
"""速度预算验证：对任意 ckpt（可带 refiner）测【512 口径整链时延】，与 V4 同会话交替比较。
口径：预热 45 s -> 3 轮 x 20 次，每次调用前后各一次 GPU 同步（与 使用说明.md §7 一致）。
用法: python lat_ckpt.py <ckpt> [<ckpt2> ...] [--refiner R.pt] [--unsharp 60]
"""
import argparse, os, sys, statistics, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--refiner", default="")
ap.add_argument("--unsharp", type=float, default=60.0)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()
dev = "cuda"
from diffusers import UNet2DModel


class Refiner(nn.Module):
    def __init__(self, ch, blocks, arch="default"):
        super().__init__()
        self.arch = arch
        out_ch = 12 if arch == "r256" else 3
        layers = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks):
            layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        layers += [nn.Conv2d(ch, out_ch, 3, 1, 1)]
        self.net = nn.Sequential(*layers); self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        if self.arch == "r256":
            base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
            return (base + self.ps(self.net(x))).clamp(-1, 1)
        return (x + self.net(x)).clamp(-1, 1)


class Chain(nn.Module):
    """复刻 opt11_export_combined 的 forward（512 in -> 512 out），用于时延对照。"""
    def __init__(self, unet, ref, sq_a, sq_1a, unsharp_pct):
        super().__init__()
        self.unet = unet; self.ref = ref; self.unsharp_pct = float(unsharp_pct)
        self.sq_a = float(sq_a); self.sq_1a = float(sq_1a)

    def _unsharp(self, t):
        sigma, percent, thr_u8 = 1.2, self.unsharp_pct, 3.0
        if percent <= 0:
            return t
        ksize = int(round(sigma * 3)) * 2 + 1
        ar = torch.arange(ksize, dtype=torch.float64, device=t.device) - (ksize - 1) / 2
        k = torch.exp(-(ar * ar) / (2.0 * sigma * sigma)); k = (k / k.sum()).to(t.dtype)
        pad = ksize // 2; c = 3
        y = F.pad(t, (pad, pad, 0, 0), mode="reflect")
        y = F.conv2d(y, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        y = F.pad(y, (0, 0, pad, pad), mode="reflect")
        blur = F.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
        diff = t - blur
        mask = (diff.abs() > thr_u8 / 255.0 * 2.0).to(t.dtype)
        out = (t + (percent / 100.0) * diff * mask).clamp(-1.0, 1.0)
        return (torch.round((out + 1.0) * 127.5) / 127.5) - 1.0

    def forward(self, x):
        xs = F.interpolate(x, size=(256, 256), mode="bicubic", align_corners=False)
        pred = self.unet(xs, torch.full((1,), 500, dtype=torch.long, device=x.device)).sample
        y = (xs - self.sq_1a * pred) / self.sq_a
        out = F.interpolate(y, size=(512, 512), mode="bicubic", align_corners=False)
        if self.ref is not None:
            out = self.ref(out)
        return self._unsharp(out)


ref = None
if a.refiner:
    rd = torch.load(a.refiner, map_location="cpu", weights_only=False)
    ref = Refiner(rd["ch"], rd["blocks"], rd.get("arch", "default")).to(dev).eval().half()
    ref.load_state_dict(rd["net"])
    print("[refiner] %s ch=%d blocks=%d steps=%s" % (Path(a.refiner).name, rd["ch"], rd["blocks"], rd.get("steps")), flush=True)

chains, names = [], []
for ck in a.ckpts:
    d = torch.load(ck, map_location="cpu", weights_only=False)
    ch = tuple(d["ch"]); res = int(d["res"]); alpha = float(d["alpha"])
    unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                       block_out_channels=ch, down_block_types=("DownBlock2D",) * len(ch),
                       up_block_types=("UpBlock2D",) * len(ch))
    unet.load_state_dict(d["unet"]); unet = unet.to(dev).eval().half()
    m = Chain(unet, ref, alpha ** 0.5, (1 - alpha) ** 0.5, a.unsharp).to(dev).eval()
    m = torch.jit.trace(m, torch.zeros(1, 3, 512, 512, device=dev, dtype=torch.float16), strict=False)
    chains.append(m); names.append("%s(ch=%s,%.2fM)" % (Path(ck).stem, ",".join(map(str, ch)),
                                                        sum(p.numel() for p in unet.parameters()) / 1e6))
    print("[load] %-42s" % names[-1], flush=True)

x = (torch.rand(1, 3, 512, 512, device="cuda") * 2 - 1).half().contiguous()
with torch.no_grad():
    t0 = time.time()
    while time.time() - t0 < 45.0:
        for m in chains:
            m(x)
    torch.cuda.synchronize()
print("[preheat] %.0f s" % (time.time() - t0), flush=True)

ts = {n: [] for n in names}
with torch.no_grad():
    for _ in range(a.runs):
        for n, m in zip(names, chains):
            for _ in range(a.iters):
                torch.cuda.synchronize(); t = time.perf_counter()
                m(x)
                torch.cuda.synchronize(); ts[n].append((time.perf_counter() - t) * 1e3)
print("\n%-44s %9s %9s %9s" % ("chain", "median", "min", "p90"))
for n in names:
    s = sorted(ts[n])
    print("%-44s %9.2f %9.2f %9.2f" % (n, statistics.median(s), s[0], s[int(0.9 * len(s))]))
K, C, P, B = 285.6, 2.515, 0.198, 0.78
print("\n%-44s %12s %10s" % ("chain", "factor", "相对基准"))
base = statistics.median(ts[names[0]]) if names else 1
for n in names:
    t = statistics.median(ts[n]); f = (K / (t + C) / B) ** P
    print("%-44s %12.4f %9.2f%%" % (n, f, (f / ((K / (base + C) / B) ** P) - 1) * 100))
print("LAT_CKPT_DONE")
