# -*- coding: utf-8 -*-
"""S1 诊断：① 模型能否 eager 跑通（绕开 CUDA graph）；② U 学到了什么（与 bicubic 的偏差）。"""
import os, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
dev = "cuda"
CK = r"D:\models\refine_v1\upref_up1b_s6000.pt"
STU = r"D:\models\tiny_v1\v21_s21000.pt"
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")

# 1) 先单独测：能否 jit.load 并 eager 前向
print("[1] jit.load + eager forward", flush=True)
m = torch.jit.load(r"D:\models\pkg_s1\your_model.pt", map_location=dev); m.eval()
x = torch.zeros(1, 3, 512, 512, device=dev, dtype=torch.float16)
with torch.no_grad():
    y = m(x)
print("    OK out=%s nan=%s range=[%.3f,%.3f]" % (tuple(y.shape), bool(torch.isnan(y).any()),
                                                  float(y.min()), float(y.max())), flush=True)

# 2) U 学到了什么：与 bicubic 上采样的偏差
print("\n[2] U 的行为", flush=True)


class Up2(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.c1 = nn.Conv2d(3, ch, 3, 1, 1); self.act = nn.ReLU(True)
        self.c2 = nn.Conv2d(ch, 12, 3, 1, 1); self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        return (base + self.ps(self.c2(self.act(self.c1(x))))).clamp(-1, 1)


sd = torch.load(CK, map_location="cpu", weights_only=False)
U = Up2(16).to(dev).eval().half(); U.load_state_dict(sd["U"])
print("    c2.weight |max| = %.5f  (zero-init => 偏离 bicubic 的程度)"
      % float(U.c2.weight.abs().max()), flush=True)
print("    c2.bias   |max| = %.5f" % float(U.c2.bias.abs().max()), flush=True)

rng = np.random.default_rng(0)
low = torch.from_numpy(rng.random((1, 3, 256, 256)).astype(np.float32) * 2 - 1).to(dev).half()
with torch.no_grad():
    up = U(low)
    bic = F.interpolate(low, scale_factor=2, mode="bicubic", align_corners=False)
d = (up.float() - bic.float()).abs()
print("    U vs bicubic: max=%.4f mean=%.5f  (0 = 完全没学动)" % (float(d.max()), float(d.mean())), flush=True)

# 3) 与现役 V4 的 512 输出对比（同一输入）
print("\n[3] 与现役 V4 包在同一 512 输入上的输出差异", flush=True)
img = np.asarray(Image.open(TEST / "case3.jpg").convert("RGB"), dtype=np.uint8)[:512, :512]
t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div_(127.5).sub_(1.0)
import importlib.util
spec = importlib.util.spec_from_file_location("r0", r"D:\models\finals_20260927\Track2_light_diffusion_inference\model_dir\runner.py")
m0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m0)
R0 = m0.Runner(r"D:\models\finals_20260927\Track2_light_diffusion_inference\model_dir")
R0.graph = None
with torch.no_grad():
    a = R0.infer(t.clone(), prompt="").float().cpu().numpy()
    b = m(t.to(dev).half()).float().cpu().numpy()
dd = np.abs(a - b)
print("    V4 vs S1: max=%.5f mean=%.6f  (float, [-1,1] 域)" % (dd.max(), dd.mean()), flush=True)
print("    V4 输出 mean=%.5f  S1 输出 mean=%.5f" % (a.mean(), b.mean()), flush=True)
print("S1_DIAG_DONE", flush=True)
