"""用候选退化配置从 3 张官方 GT 合成 LQ, 与官方 LQ 的无参统计对比 (验收校准)。

用法: python tmp_synth_lq_check.py [out_dir]
随后: python eval.py --pred-dir <out_dir> --pred-pattern '*.png' --out <out>.json
对比官方 LQ 基线: clipiqa 0.251 / musiq 24.09 / maniqa 0.302 / niqe 7.67
"""
import io
import sys
import random
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

VAL = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\验证集")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tmp_synth_lq")


def _jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_jbn(img, blur_range, noise_range, jpeg_range):
    """JPEG -> blur -> noise (case1/2 的拟合顺序), 同分辨率。"""
    q = random.randint(*jpeg_range)
    img = _jpeg(img, q)
    r = random.uniform(*blur_range)
    if r > 0:
        img = img.filter(ImageFilter.GaussianBlur(r))
    if noise_range[1] > 0:
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + np.random.randn(*arr.shape) * random.uniform(*noise_range)
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = dict(blur_range=(0.3, 2.2), noise_range=(0.0, 1.2), jpeg_range=(55, 95))
    for case in (1, 2, 3):
        gt_p = VAL / f"case{case}_gt.png"
        if not gt_p.exists():
            gt_p = VAL / f"case{case}_gt.jpg"
        gt = Image.open(gt_p).convert("RGB")
        lq = degrade_jbn(gt, **cfg)
        lq.save(OUT / f"case{case}.png")
        print(f"case{case}: synthetic LQ saved")
    print("cfg:", cfg)
    print("now run: python eval.py --pred-dir tmp_synth_lq --pred-pattern '*.png' --out results/tmp_synth_lq.json")
