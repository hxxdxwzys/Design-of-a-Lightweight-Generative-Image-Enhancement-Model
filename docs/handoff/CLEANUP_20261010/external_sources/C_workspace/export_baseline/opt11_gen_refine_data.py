r"""生成 refiner 训练输入: v21 学生在 distill256/lq 上的 512 域输出（与学生渲染路径完全一致: down2 -> UNet -> up2）。"""
import os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

CKPT = r"D:\models\tiny_v1\v21_s21000.pt"
SRC = Path(r"D:\datasets\distill256\lq")
OUT = Path(r"D:\models\refine_v1\in")
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
from diffusers import UNet2DModel
d = torch.load(CKPT, map_location="cpu", weights_only=False)
ch = tuple(d["ch"]); res = int(d["res"]); alpha = float(d["alpha"])
unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=ch, down_block_types=("DownBlock2D",)*len(ch),
                   up_block_types=("UpBlock2D",)*len(ch))
unet.load_state_dict(d["unet"]); net = unet.to(dev).eval().half()
sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5
fs = sorted(SRC.glob("*.jpg"))
print("待处理", len(fs), flush=True)
t0 = time.time()
with torch.no_grad():
    for i, f in enumerate(fs, 1):
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        x = torch.from_numpy(a).permute(2, 0, 1)[None].to(dev).float().div_(127.5).sub_(1)
        xh = x.half()
        xs = F.interpolate(xh, size=(256, 256), mode="bicubic", align_corners=False)
        pred = net(xs, torch.full((1,), 500, dtype=torch.long, device=dev)).sample
        y = (xs - sq_1a * pred) / sq_a
        y = F.interpolate(y, size=(512, 512), mode="bicubic", align_corners=False).float().clamp(-1, 1)
        o = ((y[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(o).save(OUT / (f.stem + ".jpg"), quality=98, subsampling=0)
        if i % 500 == 0:
            print("  %d/%d  %.2f s/张" % (i, len(fs), (time.time()-t0)/i), flush=True)
print("GEN_REFINE_DATA_DONE ->", OUT, flush=True)
