# -*- coding: utf-8 -*-
"""S1 导出器：把 (冻结学生 + 可学习上采样 U + refiner G + 内置 unsharp) 导出为
与现役提交包【同接口】的 model_dir/{your_model.pt, runner.py}，从而复用已验证的渲染/评测链。

推理链（与 S1 训练时逐项一致）：
  512 in -> 双三次↓256 -> 学生一步扩散 -> 双三次↑512 -> U↑512(残差) -> G refiner -> unsharp(thr)

用法: python s1_export.py <upref.pt> <modeldir> [--thr 1.0]
"""
import argparse, os, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("modeldir")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--thr", type=float, default=1.0)
ap.add_argument("--sigma", type=float, default=1.2)
ap.add_argument("--pct", type=float, default=60.0)
a = ap.parse_args()
dev = "cuda"
from diffusers import UNet2DModel

STU = r"D:\models\tiny_v1\v21_s21000.pt"


class Up2(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.c1 = nn.Conv2d(3, ch, 3, 1, 1); self.act = nn.ReLU(True)
        self.c2 = nn.Conv2d(ch, 12, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        return (base + self.ps(self.c2(self.act(self.c1(x))))).clamp(-1, 1)


class Refiner(nn.Module):
    def __init__(self, ch, blocks):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks):
            layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        layers += [nn.Conv2d(ch, 3, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return (x + self.net(x)).clamp(-1, 1)


d = torch.load(STU, map_location="cpu", weights_only=False)
ch_s = tuple(d["ch"]); res = int(d["res"]); alpha = float(d["alpha"])
unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=ch_s, down_block_types=("DownBlock2D",) * len(ch_s),
                   up_block_types=("UpBlock2D",) * len(ch_s))
unet.load_state_dict(d["unet"]); unet = unet.to(dev).eval().half()
sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5

sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
U = Up2(16).to(dev).eval().half(); U.load_state_dict(sd["U"])
G = Refiner(sd["ch"], sd["blocks"]).to(dev).eval().half(); G.load_state_dict(sd["net"])
print("[s1] upref=%s steps=%s U=%d G=%d" % (Path(a.ckpt).name, sd.get("steps"),
      sum(p.numel() for p in U.parameters()), sum(p.numel() for p in G.parameters())), flush=True)


class Combined(nn.Module):
    def __init__(self, unet, U, G, sq_a, sq_1a, pct, sigma, thr, size=512):
        super().__init__()
        self.unet = unet; self.U = U; self.G = G
        self.pct = float(pct); self.sigma = float(sigma); self.thr = float(thr); self.size = int(size)
        self.register_buffer("sq_a", torch.tensor(float(sq_a)))
        self.register_buffer("sq_1a", torch.tensor(float(sq_1a)))
        self.register_buffer("tt", torch.tensor([500], dtype=torch.long))

    def _unsharp(self, t):
        sigma, percent, thr_u8 = float(self.sigma), float(self.pct), float(self.thr)
        ksize = int(round(sigma * 3)) * 2 + 1
        ar = torch.arange(ksize, dtype=torch.float64, device=t.device) - (ksize - 1) / 2
        k = torch.exp(-(ar * ar) / (2.0 * sigma * sigma)); k = (k / k.sum()).to(t.dtype)
        pad = ksize // 2; c = 3
        y = F.pad(t, (pad, pad, 0, 0), mode="reflect")
        y = F.conv2d(y, k.reshape(1, 1, 1, ksize).expand(c, 1, 1, ksize), groups=c)
        y = F.pad(y, (0, 0, pad, pad), mode="reflect")
        blur = F.conv2d(y, k.reshape(1, 1, ksize, 1).expand(c, 1, ksize, 1), groups=c)
        diff = t - blur
        mask = (diff.abs() > thr_u8 / 255.0 * 2.0).to(t.dtype)
        out = (t + (percent / 100.0) * diff * mask).clamp(-1.0, 1.0)
        return (torch.round((out + 1.0) * 127.5) / 127.5) - 1.0

    def forward(self, x):
        # 与 S1 训练链逐项一致：512 ->(双三次↓256)-> 学生 ->(双三次↑512)-> U↑512 -> refiner@512
        S = int(self.size)
        xs = F.interpolate(x, size=(S, S), mode="bicubic", align_corners=False)
        pred = self.unet(xs, self.tt).sample
        y = (xs - self.sq_1a.to(x.dtype) * pred) / self.sq_a.to(x.dtype)   # 学生输出，S 域
        return self._unsharp(self.G(self.U(y)))


M = Combined(unet, U, G, sq_a, sq_1a, a.pct, a.sigma, a.thr, a.size).to(dev).eval().half()
dummy = torch.zeros(1, 3, 512, 512, device=dev, dtype=torch.float16)
with torch.no_grad():
    M = torch.jit.trace(M, dummy, strict=False)
    out = M(dummy)
print("[trace] out=%s nan=%s" % (tuple(out.shape), bool(torch.isnan(out).any())), flush=True)

md = Path(a.modeldir); md.mkdir(parents=True, exist_ok=True)
M.save(str(md / "your_model.pt"))
RUNNER = Path(r"D:\models\finals_20260927\Track2_light_diffusion_inference\model_dir\runner.py").read_text(encoding="utf-8")
(md / "runner.py").write_text(RUNNER, encoding="utf-8")
print("[export] %s  your_model.pt=%.2fMB" % (md, (md / "your_model.pt").stat().st_size / 2**20), flush=True)
print("S1_EXPORT_DONE", flush=True)
