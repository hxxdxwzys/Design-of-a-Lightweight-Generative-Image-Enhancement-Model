"""把 HYPIR 2x 输出 (8K) 降采样回原始 4K 尺寸, 供 eval 对比。"""
import re
from PIL import Image
from pathlib import Path

VAL = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\验证集")
SRC = Path(r"D:/models/hypir/out_val3_2x/result")
OUT = Path(r"D:/models/hypir/out_val3_2x/result_1x")
OUT.mkdir(parents=True, exist_ok=True)

sizes = {}
for case in (1, 2, 3):
    sizes[case] = Image.open(VAL / f"case{case}_lq.jpg").size
print("original LQ sizes:", sizes)

for src in sorted(SRC.glob("*.png")):
    m = re.search(r"case(\d+)", src.name)
    if not m:
        continue
    case = int(m.group(1))
    img = Image.open(src).convert("RGB")
    print(f"case{case}: {img.size} -> {sizes[case]}")
    img = img.resize(sizes[case], Image.LANCZOS)
    img.save(OUT / f"case{case}.png")
print("saved to", OUT)
