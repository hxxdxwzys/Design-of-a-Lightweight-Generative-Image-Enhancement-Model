# -*- coding: utf-8 -*-
"""架构上限量化 (2026-09-08): qwen VAE 往返重建 lpips = 1.4B 家族的理论 lpips 地板?
若 VAE 往返误差 ≈ 0.35, 则 m0s1 的 0.357 已近触底 -> 架构性证明
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:/models/finals/VOSR')
sys.path.insert(0, r'D:/models/finals/VOSR/models')

QWEN = r'C:/models/finals/preset/ckpts/Qwen-Image-vae-2d'
GT_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')


def main():
    import pyiqa
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    dev = 'cuda'
    vae = AutoencoderKLQwenImage2D.from_pretrained(QWEN).to(dev).half().eval()
    lpips_m = pyiqa.create_metric('lpips', device=dev)
    lm = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1).to(dev).half()
    lsi = (1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)).to(dev).half()

    gts = sorted(GT_DIR.glob('*_gt.*'))
    print(f'GT 图: {len(gts)}')
    vals = []
    for g in gts:
        img = Image.open(g).convert('RGB')
        x = torch.from_numpy(np.asarray(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
        x = x.to(dev).half()
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode()
            rec = vae.decode(z, return_dict=False)[0]
        # lpips 在 fp32 上算
        rec01 = ((rec.float() + 1) / 2).clamp(1e-4, 1 - 1e-4)
        gt01 = ((x.float() + 1) / 2).clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            v = lpips_m(rec01, gt01).item()
        vals.append(v)
        print(f'{g.name}: VAE 往返 lpips = {v:.4f}', flush=True)
    print(f'\n均值: {sum(vals)/len(vals):.4f}')
    print(f'对比: m0s1-sc0.93 全链 lpips = 0.3571 (差值 = 模型增强贡献)')
    print(f'对比: M24 全链 lpips = 0.3852')


if __name__ == '__main__':
    main()
