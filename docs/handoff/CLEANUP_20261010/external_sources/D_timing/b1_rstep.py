# -*- coding: utf-8 -*-
"""B1（回归⑩）：refiner 多步（步数 ≠ 容量）。
审计已证 refiner【容量】(16→32ch)无效、训练【步数】无效、【训练域】无效、【损失配方】无效；
但从未测过【推理时的迭代步数】—— 同一组 11K 参数重复施加残差，等价于更深/更强的细节合成器。

两种施加方式，都无需重训（直接用已提交 V4 的 refiner 权重）：
  iter   : x <- R(x) 迭代 n 次（复合）
  resid  : x + n*(R(x)-x)（残差放大）
时延代价：每多一遍 ≈ +2.1 ms（refiner 单遍实测）。

预登记判据（跑完不改）：全链 clipiqa 相对 V4(0.8042)/V4t1(0.8118)
  >= 0.83      => 活
  > 0.8118     => 正向，但须扣时延代价后判净值
  <= 0.8042    => 死
用法: python b1_rstep.py <mode> <n> <tag>
"""
import os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
TILE, OVER = 512, 32
mode = sys.argv[1]; NSTEP = int(sys.argv[2]); TAG = sys.argv[3]
dev = "cuda"
from diffusers import UNet2DModel

STU = r"D:\models\tiny_v1\v21_s21000.pt"
REF = r"D:\models\refine_v1\refiner_thin2_s6000.pt"


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


def unsharp(t, percent=60.0, sigma=1.2, thr_u8=1.0):
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


d = torch.load(STU, map_location="cpu", weights_only=False)
ch = tuple(d["ch"]); res = int(d["res"]); alpha = float(d["alpha"])
unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=ch, down_block_types=("DownBlock2D",) * len(ch),
                   up_block_types=("UpBlock2D",) * len(ch))
unet.load_state_dict(d["unet"]); net = unet.to(dev).eval().half()
rd = torch.load(REF, map_location="cpu", weights_only=False)
ref = Refiner(rd["ch"], rd["blocks"], rd.get("arch", "default")).to(dev).eval().half()
ref.load_state_dict(rd["net"])
sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5
print("[b1] mode=%s n=%d tag=%s | refiner ch=%d blocks=%d" % (mode, NSTEP, TAG, rd["ch"], rd["blocks"]), flush=True)

mask = np.ones((TILE, TILE, 1), dtype=np.float32); r = OVER // 2
ramp = np.linspace(0, 1, r, dtype=np.float32)
mask[:r, :, 0] *= ramp[:, None]; mask[-r:, :, 0] *= ramp[::-1, None]
mask[:, :r, 0] *= ramp[None, :]; mask[:, -r:, 0] *= ramp[::-1][None, :]


@torch.no_grad()
def run(patch_u8):
    t = torch.from_numpy(np.ascontiguousarray(patch_u8)).permute(2, 0, 1)[None].to(dev).float().div_(127.5).sub_(1)
    xin = F.interpolate(t.half(), size=(res, res), mode="bicubic", align_corners=False)
    pred = net(xin, torch.full((1,), 500, dtype=torch.long, device=dev)).sample
    y = (xin - sq_1a * pred) / sq_a
    y = F.interpolate(y, size=(TILE, TILE), mode="bicubic", align_corners=False)
    base = y
    if mode == "iter":
        for _ in range(NSTEP):
            y = ref(y)
    else:  # resid 放大
        y = (base + NSTEP * (ref(base) - base)).clamp(-1, 1)
    y = unsharp(y)
    o = y[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip((o + 1) * 127.5, 0, 255)


out = Path(r"D:\models\opt11") / ("eval_" + TAG); out.mkdir(parents=True, exist_ok=True)
files = sorted(TEST.glob("*.jpg"))[:30]
t0 = time.time()
for i, p in enumerate(files, 1):
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    H, W = img.shape[:2]
    acc = np.zeros((H, W, 3), dtype=np.float32); wsum = np.zeros((H, W, 1), dtype=np.float32)
    ys = sorted(set(list(range(0, max(1, H - TILE + 1), TILE - OVER)) + [max(0, H - TILE)]))
    xs = sorted(set(list(range(0, max(1, W - TILE + 1), TILE - OVER)) + [max(0, W - TILE)]))
    for y0 in ys:
        for x0 in xs:
            acc[y0:y0 + TILE, x0:x0 + TILE] += run(img[y0:y0 + TILE, x0:x0 + TILE]) * mask
            wsum[y0:y0 + TILE, x0:x0 + TILE] += mask
    Image.fromarray(np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)).save(out / p.name, quality=98, subsampling=0)
    if i % 10 == 0 or i == len(files):
        print("   %d/%d %.2f s/张" % (i, len(files), (time.time() - t0) / i), flush=True)
print("B1_RENDER_DONE", TAG, flush=True)
