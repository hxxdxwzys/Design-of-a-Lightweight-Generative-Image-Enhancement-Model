# -*- coding: utf-8 -*-
"""B2 评测器：渲染带【可学习下采样器】的学生（把 DN 嵌进链路），口径与历史 test30 完全一致。
用法: python b2_eval.py <ckpt> <tag> [--n 30]
判据（预登记）：学生单独 clipiqa@4K  vs v21 的 0.6306
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
TILE, OVER = 512, 32

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("tag")
ap.add_argument("--n", type=int, default=30)
a = ap.parse_args()
dev = "cuda"
from diffusers import UNet2DModel


class LearnDown(nn.Module):
    def __init__(self, ch=3):
        super().__init__()
        ar = torch.arange(5, dtype=torch.float32) - 2.0
        k = torch.exp(-(ar * ar) / 2.0)
        k = (k / k.sum()).view(1, 1, 1, 5).repeat(ch, 1, 1, 1)
        self.k = nn.Parameter(k.clone())

    def forward(self, x):
        c = x.shape[1]
        w = self.k[:c]
        w = w / w.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        y = F.pad(x, (2, 2, 0, 0), mode="reflect")
        y = F.conv2d(y, w, groups=c)
        y = F.pad(y, (0, 0, 2, 2), mode="reflect")
        y = F.conv2d(y, w.transpose(2, 3).contiguous(), groups=c)
        return y[:, :, ::2, ::2]


d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
ch = tuple(d["ch"]); res = int(d["res"]); alpha = float(d["alpha"])
use_dn = bool(d.get("lrn_down")) and ("down" in d)
print("[b2] ckpt=%s ch=%s res=%d steps=%s lrn_down=%s" % (Path(a.ckpt).name, ch, res, d.get("steps"), use_dn), flush=True)
unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=ch, down_block_types=("DownBlock2D",) * len(ch),
                   up_block_types=("UpBlock2D",) * len(ch))
unet.load_state_dict(d["unet"]); net = unet.to(dev).eval().half()
DN = None
if use_dn:
    DN = LearnDown().to(dev).eval().half(); DN.load_state_dict(d["down"])
    print("[b2] LearnDown kernel (after norm): %s" % DN.k.detach().float().flatten()[:5].tolist(), flush=True)
sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5

out = Path(r"D:\models\opt11") / ("eval_" + a.tag)
out.mkdir(parents=True, exist_ok=True)
mask = np.ones((TILE, TILE, 1), dtype=np.float32); r = OVER // 2
ramp = np.linspace(0, 1, r, dtype=np.float32)
mask[:r, :, 0] *= ramp[:, None]; mask[-r:, :, 0] *= ramp[::-1, None]
mask[:, :r, 0] *= ramp[None, :]; mask[:, -r:, 0] *= ramp[::-1][None, :]


@torch.no_grad()
def run(patch_u8):
    t = torch.from_numpy(np.ascontiguousarray(patch_u8)).permute(2, 0, 1)[None].to(dev).float().div_(127.5).sub_(1)
    if DN is not None:
        xin = DN(t.half())                     # 可学习抗锯齿下采样（替代双三次）
    else:
        xin = F.interpolate(t.half(), size=(res, res), mode="bicubic", align_corners=False)
    pred = net(xin, torch.full((1,), 500, dtype=torch.long, device=dev)).sample
    y = (xin - sq_1a * pred) / sq_a
    y = F.interpolate(y, size=(TILE, TILE), mode="bicubic", align_corners=False)
    o = y[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip((o + 1) * 127.5, 0, 255)


files = sorted(TEST.glob("*.jpg"))[:a.n]
print("[b2] %d 张 -> %s" % (len(files), out), flush=True)
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
    res_img = np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)
    Image.fromarray(res_img).save(out / p.name, quality=98, subsampling=0)
    if i % 10 == 0 or i == len(files):
        print("   %d/%d %.2f s/张" % (i, len(files), (time.time() - t0) / i), flush=True)
print("B2_EVAL_RENDER_DONE ->", out, flush=True)
