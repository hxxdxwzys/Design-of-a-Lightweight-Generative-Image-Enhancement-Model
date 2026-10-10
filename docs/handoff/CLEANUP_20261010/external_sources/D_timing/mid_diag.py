# -*- coding: utf-8 -*-
"""中间量对比：V4 与 S1b 在同一输入下逐段比较，找出偏差出现的位置。"""
import os, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
dev = "cuda"
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
img = np.asarray(Image.open(TEST / "case3.jpg").convert("RGB"), dtype=np.uint8)[:512, :512]
t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div_(127.5).sub_(1.0)
x = t.to(dev).half()
print("input range=[%.3f, %.3f]" % (float(x.min()), float(x.max())), flush=True)

# --- V4 ---
m4 = torch.jit.load(r"D:\models\finals_20260927\Track2_light_diffusion_inference\model_dir\your_model.pt",
                    map_location=dev); m4.eval()
with torch.no_grad():
    y4 = m4(x)
print("\nV4    out range=[%.3f, %.3f]  out.min 处像素数=%d" % (
    float(y4.min()), float(y4.max()), int((y4 <= -0.999).sum())), flush=True)

# --- S1b 组件重组（在 Python 里逐步算，看哪一段出问题）---
from diffusers import UNet2DModel
STU = r"D:\models\tiny_v1\v21_s21000.pt"
d = torch.load(STU, map_location="cpu", weights_only=False)
ch_s = tuple(d["ch"]); res = int(d["res"]); alpha = float(d["alpha"])
unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=ch_s, down_block_types=("DownBlock2D",) * len(ch_s),
                   up_block_types=("UpBlock2D",) * len(ch_s))
unet.load_state_dict(d["unet"]); unet = unet.to(dev).eval().half()
sq_a, sq_1a = alpha ** 0.5, (1 - alpha) ** 0.5

with torch.no_grad():
    xs = F.interpolate(x, size=(256, 256), mode="bicubic", align_corners=False)
    pred = unet(xs, torch.full((1,), 500, dtype=torch.long, device=dev)).sample
    y = (xs - sq_1a * pred) / sq_a
    print("\n[student] pred range=[%.3f,%.3f]  x0 range=[%.3f,%.3f]  (>1 的占比 %.3f%%)" % (
        float(pred.min()), float(pred.max()), float(y.min()), float(y.max()),
        float((y.abs() > 1).float().mean() * 100)), flush=True)
    y512 = F.interpolate(y, size=(512, 512), mode="bicubic", align_corners=False)
    print("[up to 512] range=[%.3f,%.3f]" % (float(y512.min()), float(y512.max())), flush=True)

# S1b 的 U/G
sd = torch.load(r"D:\models\refine_v1\uident_s1b_s2000.pt", map_location="cpu", weights_only=False)


class UIdent(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.c1 = nn.Conv2d(3, ch, 3, 1, 1); self.act = nn.ReLU(True)
        self.c2 = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, x):
        return (x + self.c2(self.act(self.c1(x)))).clamp(-1, 1)


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


U = UIdent(32).to(dev).eval().half(); U.load_state_dict(sd["U"])
G = Refiner(sd["ch"], sd["blocks"]).to(dev).eval().half(); G.load_state_dict(sd["net"])
with torch.no_grad():
    yu = U(y512)
    print("[U out] range=[%.3f,%.3f]  (U 偏离恒等 max=%.5f)" % (
        float(yu.min()), float(yu.max()), float((yu - y512).abs().max())), flush=True)
    yg = G(yu)
    print("[G out] range=[%.3f,%.3f]" % (float(yg.min()), float(yg.max())), flush=True)

# V4 的 refiner 在同输入上的输出（用于对照）
rd = torch.load(r"D:\models\refine_v1\refiner_thin2_s6000.pt", map_location="cpu", weights_only=False)
G4 = Refiner(rd["ch"], rd["blocks"]).to(dev).eval().half(); G4.load_state_dict(rd["net"])
with torch.no_grad():
    yg4 = G4(y512)
    print("[V4 refiner on same input] range=[%.3f,%.3f]" % (float(yg4.min()), float(yg4.max())), flush=True)
    dd = (yg - yg4).abs()
    print("[G(S1b) vs G4] max=%.5f mean=%.6f" % (float(dd.max()), float(dd.mean())), flush=True)
print("MID_DIAG_DONE", flush=True)
