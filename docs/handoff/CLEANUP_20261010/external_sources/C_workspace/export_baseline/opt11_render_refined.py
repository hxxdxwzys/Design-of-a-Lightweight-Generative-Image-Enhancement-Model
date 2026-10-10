r"""用 (学生 + 细节合成器) 渲染测试集 4K，口径与 render_tiny 一致（512 块, overlap 32, 羽化, q98）。
用法: python opt11_render_refined.py --ckpt X.pt --refiner R.pt --out DIR [--n 30 --order lex]
"""
import argparse, os, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
TILE, OVER = 512, 32
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--refiner", default="")
ap.add_argument("--out", required=True); ap.add_argument("--down", type=int, default=2)
ap.add_argument("--n", type=int, default=0); ap.add_argument("--order", default="lex")
ap.add_argument("--unsharp", type=float, default=0.0, help="轻度 unsharp 百分比（与最终 runner 一致；0=关闭）")
a = ap.parse_args()
dev = "cuda"

class Refiner(nn.Module):
    def __init__(self, ch, blocks, arch="default"):
        super().__init__()
        self.arch = arch
        out_ch = 12 if arch == "r256" else 3
        layers = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks): layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        layers += [nn.Conv2d(ch, out_ch, 3, 1, 1)]
        self.net = nn.Sequential(*layers)
        self.ps = nn.PixelShuffle(2)
    def forward(self, x):
        if self.arch == "r256":
            base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
            return (base + self.ps(self.net(x))).clamp(-1, 1)
        return (x + self.net(x)).clamp(-1, 1)

from diffusers import UNet2DModel
d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
ch = tuple(d["ch"]); res = int(d["res"]); alpha = float(d["alpha"])
unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=ch, down_block_types=("DownBlock2D",)*len(ch), up_block_types=("UpBlock2D",)*len(ch))
unet.load_state_dict(d["unet"]); net = unet.to(dev).eval().half()
sq_a = alpha**0.5; sq_1a = (1-alpha)**0.5
ref = None
if a.refiner:
    rd = torch.load(a.refiner, map_location="cpu", weights_only=False)
    _arch = rd.get("arch", "default")
    ref = Refiner(rd["ch"], rd["blocks"], _arch).to(dev).eval().half()
    ref.load_state_dict(rd["net"])
    print("[refiner] arch=%s ch=%d blocks=%d steps=%s" % (_arch, rd["ch"], rd["blocks"], rd.get("steps")), flush=True)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
mask = np.ones((TILE, TILE, 1), dtype=np.float32); r = OVER // 2
ramp = np.linspace(0, 1, r, dtype=np.float32)
mask[:r, :, 0] *= ramp[:, None]; mask[-r:, :, 0] *= ramp[::-1, None]
mask[:, :r, 0] *= ramp[None, :]; mask[:, -r:, 0] *= ramp[::-1][None, :]

def _unsharp(t, percent, sigma=1.2, thr_u8=3.0):
    if percent <= 0:
        return t
    ksize = int(round(sigma * 3)) * 2 + 1
    ar = torch.arange(ksize, dtype=torch.float64, device=t.device) - (ksize - 1) / 2
    k = torch.exp(-(ar * ar) / (2.0 * sigma * sigma)); k = (k / k.sum()).to(t.dtype)
    pad = ksize // 2; c = int(t.shape[1])
    y = F.pad(t, (pad, pad, 0, 0), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    y = F.pad(y, (0, 0, pad, pad), mode="reflect")
    blur = F.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    diff = t - blur
    mask = (diff.abs() > thr_u8 / 255.0 * 2.0).to(t.dtype)
    out = (t + (percent / 100.0) * diff * mask).clamp(-1.0, 1.0)
    return (torch.round((out + 1.0) * 127.5) / 127.5) - 1.0


@torch.no_grad()
def run(patch_u8):
    t = torch.from_numpy(patch_u8).permute(2, 0, 1)[None].to(dev).float().div_(127.5).sub_(1)
    th = t.half()
    if a.down > 1:
        xs = F.interpolate(th, size=(TILE // a.down, TILE // a.down), mode="bicubic", align_corners=False)
    else:
        xs = th
    pred = net(xs, torch.full((1,), 500, dtype=torch.long, device=dev)).sample
    y = (xs - sq_1a * pred) / sq_a
    if ref is not None and getattr(ref, "arch", "default") == "r256":
        y = ref(y.half())                      # refiner 内部完成 2x 上采样
    else:
        if a.down > 1:
            y = F.interpolate(y, size=(TILE, TILE), mode="bicubic", align_corners=False)
        if ref is not None:
            y = ref(y.half())
    y = _unsharp(y, a.unsharp)
    o = y[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip((o + 1) * 127.5, 0, 255)

files = sorted(TEST.glob("case*.jpg"), key=lambda p: int(p.stem[4:])) if a.order == "num" else sorted(TEST.glob("case*.jpg"))
if a.n: files = files[:a.n]
print("[render] %d 张" % len(files), flush=True)
t0 = time.time()
for i, f in enumerate(files, 1):
    img = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
    H, W = img.shape[:2]
    acc = np.zeros((H, W, 3), dtype=np.float32); wsum = np.zeros((H, W, 1), dtype=np.float32)
    ys = sorted(set(list(range(0, max(1, H-TILE+1), TILE-OVER)) + [max(0, H-TILE)]))
    xs_ = sorted(set(list(range(0, max(1, W-TILE+1), TILE-OVER)) + [max(0, W-TILE)]))
    for y0 in ys:
        for x0 in xs_:
            acc[y0:y0+TILE, x0:x0+TILE] += run(img[y0:y0+TILE, x0:x0+TILE]) * mask
            wsum[y0:y0+TILE, x0:x0+TILE] += mask
    res = np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)
    Image.fromarray(res).save(out / f.name, quality=98, subsampling=0)
    if i % 10 == 0 or i == len(files):
        print("  %d/%d %.2f s/张" % (i, len(files), (time.time()-t0)/i), flush=True)
print("RENDER_REFINED_DONE", flush=True)
