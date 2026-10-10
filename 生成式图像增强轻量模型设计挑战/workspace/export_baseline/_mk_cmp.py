import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
names = sorted(p.name for p in TEST.glob("case*.jpg"))
pick = names[9] if len(names) > 9 else names[0]
print("pick:", pick)
paths = {"LQ": TEST / pick,
         "V4": Path(r"D:\models\opt11\v4_render") / pick,
         "clip2": Path(r"D:\models\opt11\clip2_render") / pick}
arrs = {}
for k, p in paths.items():
    im = Image.open(p).convert("RGB")
    arrs[k] = np.asarray(im)
    print(k, arrs[k].shape)
h, w = arrs["LQ"].shape[:2]
C = 400
y0, x0 = (h - C)//2, (w - C)//2
tiles = [arrs[k][y0:y0+C, x0:x0+C] for k in ("LQ", "V4", "clip2")]
gap = np.full((C, 8, 3), 255, dtype=np.uint8)
row = np.concatenate([tiles[0], gap, tiles[1], gap, tiles[2]], axis=1)
out = Path(r"D:\models\opt11\cmp_lq_v4_clip2.png")
Image.fromarray(row).save(out)
print("saved", out, row.shape)