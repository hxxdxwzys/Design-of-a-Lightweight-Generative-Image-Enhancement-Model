#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出 256 内部分辨率测速模型 JIT (公式探测用): 512 输入 -> 256 增强 -> 512 输出。
runner 在 infer 内做 bicubic 缩放。时延 ~6-8ms → 加速比 ~40x。
用法: python tmp_export_probe_256.py <out.pt>
"""
import sys
import time
import torch
from diffusers import UNet2DModel, DDPMScheduler

sys.path.insert(0, r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace")


class PixelSpeedModel(torch.nn.Module):
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
    out = sys.argv[1] if len(sys.argv) > 1 else 'probe_fast_model.pt'
    device = 'cuda'
    model = PixelSpeedModel((32, 32, 32, 64)).to(device).eval().half()
    print(f'params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)
    ex = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float16).clamp(-1, 1)
    with torch.no_grad():
        traced = torch.jit.trace(model, ex)
    traced.save(out)
    with torch.no_grad():
        o = traced(ex)
    print(f'saved {out} | out {tuple(o.shape)} NaN={torch.isnan(o).any().item()}', flush=True)
    # timing at 256
    for _ in range(5):
        traced(ex)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        traced(ex)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 200 * 1000
    # timing at 512 for comparison (same model)
    ex512 = torch.nn.functional.interpolate(ex, size=(512, 512))
    for _ in range(5):
        traced(ex512)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(200):
        traced(ex512)
    torch.cuda.synchronize()
    dt512 = (time.perf_counter() - t1) / 200 * 1000
    print(f'256: {dt:.1f}ms (~{287/dt:.0f}x) | 512: {dt512:.1f}ms (~{287/dt512:.0f}x)  [基准 287ms]', flush=True)


if __name__ == '__main__':
    main()
