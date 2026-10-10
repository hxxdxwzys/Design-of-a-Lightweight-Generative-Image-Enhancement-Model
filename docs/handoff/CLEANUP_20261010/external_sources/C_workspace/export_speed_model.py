#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出极轻测速模型 JIT (TinyVAE + 轻 UNet, fp16) 用于提交测速。

测速模型不需要质量 (规则只要求 Diffusion 架构 + 快), 目标是极致时延:
  TinyVAE (27ms) + 轻 UNet 32/64/128/256 (11ms) ≈ 31ms → 加速比 9.6-12.8x

用法: python export_speed_model.py <out.pt> [--unet_ch 32,64,128,256] [--tiny_vae 本地缓存路径]
"""
import os
import sys
import argparse
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.light_model import LightLatentDiff
from diffusers import AutoencoderTiny

TAESD_DEFAULT = ('C:/Users/Yuki Nagato/.cache/huggingface/hub/'
                 'models--madebyollin--taesd/snapshots/'
                 '614f76814bbe30edbe2e627ace1c2234c81a2c0e')


class SpeedModel(torch.nn.Module):
    """TinyVAE + 轻 UNet 一步潜扩散 (测速专用, 不需高质量)。"""

    def __init__(self, unet_ch, tiny_vae_dir):
        super().__init__()
        # 复用 LightLatentDiff 的 VAE 结构但替换为 TAESD
        self.base = LightLatentDiff('D:/models/sd21base', unet_ch=unet_ch)
        self.tiny = AutoencoderTiny.from_pretrained(tiny_vae_dir,
                                                    torch_dtype=torch.float16)
        self.alpha_t = self.base.alpha_t
        self.t = torch.tensor([500], dtype=torch.long)

    def to(self, *args, **kwargs):
        """确保 alpha_t/t 也随设备移动。"""
        super().to(*args, **kwargs)
        if hasattr(self, 'alpha_t'):
            self.alpha_t = self.alpha_t.to(*args, **kwargs)
        if hasattr(self, 't'):
            self.t = self.t.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        lat = self.tiny.encode(x).latents
        pred = self.base.unet(lat, self.t).sample
        x0 = (lat - (1 - self.alpha_t).sqrt() * pred) / self.alpha_t.sqrt()
        return self.tiny.decode(x0).sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', help='输出 your_model.pt 路径')
    ap.add_argument('--unet_ch', default='32,64,128,256')
    ap.add_argument('--tiny_vae', default=TAESD_DEFAULT)
    args = ap.parse_args()

    device = 'cuda'
    unet_ch = tuple(int(x) for x in args.unet_ch.split(','))
    model = SpeedModel(unet_ch, args.tiny_vae).to(device).eval()
    # 全部 fp16
    model = model.half()
    print(f'SpeedModel 参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)

    example = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float16).clamp(-1, 1)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced.save(args.out)
    with torch.no_grad():
        o = traced(example)
    print(f'saved JIT -> {args.out} | out {tuple(o.shape)} '
          f'NaN={torch.isnan(o).any().item()}', flush=True)

    # 测时延
    import time
    for _ in range(3):
        traced(example)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 50
    for _ in range(N):
        traced(example)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N * 1000
    print(f'JIT 512 时延: {dt:.1f}ms | 加速比 300ms={300/dt:.1f}x, 400ms={400/dt:.1f}x', flush=True)


if __name__ == '__main__':
    main()
