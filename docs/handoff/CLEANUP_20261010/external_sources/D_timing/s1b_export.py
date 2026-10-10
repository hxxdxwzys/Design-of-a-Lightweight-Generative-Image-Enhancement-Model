# -*- coding: utf-8 -*-
"""S1b 导出器：学生@256->512，然后 U(可学习恒等, S域) -> refiner(S域) -> unsharp(thr=1)。
与 s1b_train 的链逐项一致；U 为恒等时精确复现已提交 V4(thr=1)。
用法: python s1b_export.py <uident.pt> <modeldir> [--thr 1.0]
"""
import argparse, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("modeldir")
ap.add_argument("--thr", type=float, default=1.0)
ap.add_argument("--sigma", type=float, default=1.2)
ap.add_argument("--pct", type=float, default=60.0)
a = ap.parse_args()
dev = "cuda"
from diffusers import UNet2DModel
STU = r"D:\models\tiny_v1\v21_s21000.pt"


class UIdent(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.c1 = nn.Conv2d(3, ch, 3, 1, 1); self.act = nn.ReLU(True)
        self.c2 = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, x):
        return x + self.c2(self.act(self.c1(x)))


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
U = UIdent(32).to(dev).eval().half(); U.load_state_dict(sd["U"])
G = Refiner(sd["ch"], sd["blocks"]).to(dev).eval().half(); G.load_state_dict(sd["net"])
print("[s1b] ckpt=%s steps=%s U=%d G=%d U|dev from ident|=%.5f" % (
    Path(a.ckpt).name, sd.get("steps"), sum(p.numel() for p in U.parameters()),
    sum(p.numel() for p in G.parameters()), float(torch.load(a.ckpt, map_location="cpu",
    weights_only=False)["U"]["c2.weight"].abs().max())), flush=True)


class Combined(nn.Module):
    def __init__(self, unet, U, G, sq_a, sq_1a, pct, sigma, thr):
        super().__init__()
        self.unet = unet; self.U = U; self.G = G
        self.pct = float(pct); self.sigma = float(sigma); self.thr = float(thr)
        self.register_buffer("sq_a", torch.tensor(float(sq_a)))
        self.register_buffer("sq_1a", torch.tensor(float(sq_1a)))
        self.register_buffer("tt", torch.tensor([500], dtype=torch.long))

    def _unsharp(self, t):
        # 关键：ksize 必须是【字面常量】，否则 trace 无法冻结形状 -> CUDA graph 捕获失败
        sigma, percent, thr_u8 = 1.2, float(self.pct), float(self.thr)
        ksize = 9
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
        xs = F.interpolate(x, size=(256, 256), mode="bicubic", align_corners=False)
        pred = self.unet(xs, self.tt).sample
        y = (xs - self.sq_1a.to(x.dtype) * pred) / self.sq_a.to(x.dtype)
        y512 = F.interpolate(y, size=(512, 512), mode="bicubic", align_corners=False)
        return self._unsharp(self.G(self.U(y512)))


M = Combined(unet, U, G, sq_a, sq_1a, a.pct, a.sigma, a.thr).to(dev).eval().half()
dummy = torch.zeros(1, 3, 512, 512, device=dev, dtype=torch.float16)
with torch.no_grad():
    M = torch.jit.trace(M, dummy, strict=False)
    out = M(dummy)
print("[trace] out=%s nan=%s" % (tuple(out.shape), bool(torch.isnan(out).any())), flush=True)
md = Path(a.modeldir); md.mkdir(parents=True, exist_ok=True)
M.save(str(md / "your_model.pt"))
(md / "runner.py").write_text(
    Path(r"D:\models\finals_20260927\Track2_light_diffusion_inference\model_dir\runner.py").read_text(encoding="utf-8"),
    encoding="utf-8")
print("[export] %s  your_model.pt=%.2fMB" % (md, (md / "your_model.pt").stat().st_size / 2**20), flush=True)
print("S1B_EXPORT_DONE", flush=True)
