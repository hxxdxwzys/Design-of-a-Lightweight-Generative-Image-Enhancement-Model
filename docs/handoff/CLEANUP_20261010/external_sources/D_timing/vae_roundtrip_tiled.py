# -*- coding: utf-8 -*-
"""VAE 往返上限 (tiled, 2026-09-08): qwen VAE 4K 往返 lpips = 架构 lpips 地板?
用 qwen_vae_lite + 1024 tile / 64 overlap 手工拼接
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r'D:/models/final_compliant_14b/model_dir')
sys.path.insert(0, r'D:/models/finals/VOSR/models')

GT_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
TILE = 1024
OVL = 64


def main():
    import json
    import pyiqa
    from PIL import Image
    from models.qwen_vae_lite import AutoencoderKLQwenImage2D
    QWEN = r'C:/models/finals/preset/ckpts/Qwen-Image-vae-2d'
    cfg = json.load(open(QWEN + '/config.json'))
    cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
    from safetensors.torch import load_file
    sd = load_file(QWEN + '/diffusion_pytorch_model.safetensors')
    sd = {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}
    dev = 'cuda'
    vae = AutoencoderKLQwenImage2D(**cfg).half().eval().to(dev)
    vae.load_state_dict(sd, strict=True)
    lpips_m = pyiqa.create_metric('lpips', device=dev)

    gts = sorted(GT_DIR.glob('*_gt.*'))
    vals = []
    for g in gts:
        img = Image.open(g).convert('RGB')
        arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev).half()
        b, c, h, w = x.shape
        rec = torch.zeros_like(x)
        step = TILE - OVL
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                y1 = min(y0 + TILE, h)
                x1 = min(x0 + TILE, w)
                y0b = max(y1 - TILE, 0)
                x0b = max(x1 - TILE, 0)
                tile = x[:, :, y0b:y1, x0b:x1]
                with torch.no_grad():
                    z = vae.encode(tile).mode()
                    r = vae.decode(z)
                # 覆盖写入 (简单起见中间区直接覆盖)
                rec[:, :, y0b:y1, x0b:x1] = r
        rec01 = ((rec.float() + 1) / 2).clamp(1e-4, 1 - 1e-4)
        gt01 = ((x.float() + 1) / 2).clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            v = lpips_m(rec01, gt01).item()
        vals.append(v)
        print(f'{g.name}: VAE 往返 lpips = {v:.4f}', flush=True)
    print(f'\n均值: {sum(vals)/len(vals):.4f}')
    print('对比: m0s1-sc0.93 全链 0.3571 | M24 全链 0.3852 | 1.4B 基座对GT 0.3322')


if __name__ == '__main__':
    main()
