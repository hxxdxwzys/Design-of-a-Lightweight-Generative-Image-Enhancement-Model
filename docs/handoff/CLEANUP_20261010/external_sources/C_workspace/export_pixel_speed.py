#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出像素域极速测速模型 JIT (无 VAE, 0.9M UNet, fp16)。

一步潜扩散: 512x512x3 像素输入 -> UNet 预测噪声 -> 显式 x0 -> 输出。
无 VAE 开销, 512 时延 ~17ms → 加速比 17-23x (vs TinyVAE 方案 30ms)。

用法: python export_pixel_speed.py <out.pt> [--channels 32,32,32,64]
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler


class PixelSpeedModel(nn.Module):
    """像素域一步潜扩散 (测速专用)。"""

    def __init__(self, channels):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=512, in_channels=3, out_channels=3,
            layers_per_block=1,
            block_out_channels=channels,
            down_block_types=('DownBlock2D',) * len(channels),
            up_block_types=('UpBlock2D',) * len(channels),
        )
        sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear',
                              beta_start=0.00085, beta_end=0.012)
        self.register_buffer('alpha_t', sched.alphas_cumprod[500].clone())
        self.register_buffer('t', torch.tensor([500], dtype=torch.long))

    def forward(self, x):
        pred = self.unet(x, self.t).sample
        x0 = (x - (1 - self.alpha_t).sqrt() * pred) / self.alpha_t.sqrt()
        return x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--channels', default='32,32,32,64')
    args = ap.parse_args()

    device = 'cuda'
    channels = tuple(int(c) for c in args.channels.split(','))
    model = PixelSpeedModel(channels).to(device).eval().half()
    print(f'PixelSpeedModel 参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)

    example = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float16).clamp(-1, 1)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced.save(args.out)
    with torch.no_grad():
        o = traced(example)
    print(f'saved JIT -> {args.out} | out {tuple(o.shape)} NaN={torch.isnan(o).any().item()}', flush=True)

    import time
    for _ in range(3):
        traced(example)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 100
    for _ in range(N):
        traced(example)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N * 1000
    print(f'JIT 512: {dt:.1f}ms | 加速比 300ms={300/dt:.1f}x, 400ms={400/dt:.1f}x', flush=True)


if __name__ == '__main__':
    main()
