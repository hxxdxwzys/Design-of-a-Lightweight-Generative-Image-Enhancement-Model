import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
names = sorted(p.name for p in TEST.glob("case*.jpg"))
pick = names[9]
dirs = {"LQ": TEST, "V4": Path(r"D:\models\opt11\v4_render"), "E8": Path(r"D:\models\opt11\e8_render"), "clip1": Path(r"D:\models\opt11\clip1_render")}
arrs = {k: np.asarray(Image.open(p/pick).convert("RGB")) for k, p in dirs.items()}
h, w = arrs["LQ"].shape[:2]; C = 340; y0, x0 = (h-C)//2, (w-C)//2
tiles = [arrs[k][y0:y0+C, x0:x0+C] for k in ("LQ","V4","E8","clip1")]
gap = np.full((C, 6, 3), 255, dtype=np.uint8)
row = tiles[0]
for t in tiles[1:]: row = np.concatenate([row, gap, t], axis=1)
out = Path(r"D:\models\opt11\cmp_lq_v4_e8_clip1.png"); Image.fromarray(row).save(out)
print("saved", out, row.shape)