"""逐图 (model_t, coeff_t) 参数扫描: 在降采样图 (最长边 ~1024) 上快速无参门控选参。

用法: python tmp_per_image_tune.py [--full]   (--full 时用 4K 跑验证)
HYPIR enhancer 只需加载一次, 通过改 model_t/coeff_t 属性复用。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import pyiqa

sys.path.insert(0, r"D:/models/hypir_code/HYPIR-main/HYPIR-main")
from HYPIR.enhancer.sd2 import SD2Enhancer

VAL = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\验证集")
BASE = r"D:/models/sd21base"
W = r"D:/models/hypir/HYPIR_sd2.pth"
MODULES = "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj".split(",")

MS = 1024  # downscale for fast gate

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true", help="4K 验证模式")
args = ap.parse_args()

device = "cuda"
enh = SD2Enhancer(base_model_path=BASE, weight_path=W, lora_modules=MODULES,
                  lora_rank=256, model_t=200, coeff_t=200, device=device)
enh.init_models()
print("model loaded")

clipiqa = pyiqa.create_metric("clipiqa", device=device)
musiq = pyiqa.create_metric("musiq", device=device)

to_tensor = transforms.ToTensor()


def run(lq_img, mt, ct):
    enh.model_t = mt
    enh.coeff_t = ct
    x = to_tensor(lq_img).unsqueeze(0).to(device)  # [0,1]
    out = enh.enhance(x, prompt="", scale_by="factor", upscale=1,
                      patch_size=512, stride=256, return_type="pt")
    return out


grids = [(150, 200), (200, 200), (250, 200), (200, 150), (200, 250), (300, 200)]

results = {}
for case in (1, 2, 3):
    lq = Image.open(VAL / f"case{case}_lq.jpg").convert("RGB")
    if not args.full:
        s = MS / max(lq.size)
        lq_small = lq.resize((round(lq.size[0] * s), round(lq.size[1] * s)), Image.BICUBIC)
    else:
        lq_small = lq
    row = []
    for mt, ct in grids:
        out = run(lq_small, mt, ct)
        c = float(clipiqa(out).mean().item())
        m = float(musiq(out).mean().item())
        row.append((mt, ct, round(c, 3), round(m, 3)))
        print(f"case{case} t={mt},{ct}: clipiqa={c:.3f} musiq={m:.3f}")
    results[case] = row

for case, row in results.items():
    best = max(row, key=lambda r: r[2] + r[3] / 50)  # 粗加权门
    print(f"case{case} best={best}")
