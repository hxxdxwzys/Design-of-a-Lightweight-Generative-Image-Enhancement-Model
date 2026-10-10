# -*- coding: utf-8 -*-
"""对照 + 组件剖析 v3 (2026-09-11)
A) 让纯 GPU 负载每次耗时 ~330ms -> 看是否出现同样的周期性尖峰（区分机器 vs 模型）
B) 逐组件计时我们的模型（防御式取 mean/std）
"""
from __future__ import annotations

import os
import statistics
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch

mode = sys.argv[1] if len(sys.argv) > 1 else 'A'


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    r = fn()
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000, r


if mode == 'A':
    a = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    print('A) 纯 GPU: 每次 20 次 4096-matmul (目标 ~300ms/次)')
    for _ in range(2):
        for _ in range(20):
            a @ b
    ts = []
    for i in range(12):
        dt, _ = timed(lambda: [a @ b for _ in range(20)])
        ts.append(dt)
    print('   ' + ' '.join(f'{v:.0f}' for v in ts))
    sp = [i for i, v in enumerate(ts) if v > 1000]
    print(f'   中位 {statistics.median(ts):.0f} ms | 尖峰 {len(sp)}/12 @ {sp}')
    print('   => 若这里也有周期性尖峰, 则是【机器/驱动】问题, 与我们模型无关')
    sys.exit(0)

# B) 组件剖析
os.chdir(r'C:/models/finals')
from pathlib import Path  # noqa: E402
md = Path(sys.argv[2])
sys.path.insert(0, str(md))
import runner as R  # noqa: E402

obj = R.Runner(str(md))
m = obj.model
print('mean 类型:', type(getattr(m, 'mean', None)),
      '| device:', getattr(getattr(m, 'mean', None), 'device', 'n/a'))
mean = getattr(m, 'mean', None)
std = getattr(m, 'std', None)
if isinstance(mean, torch.Tensor) and mean.device.type == 'cpu':
    mean = mean.to(obj.device)
    std = std.to(obj.device)
    print('  -> 已把 mean/std 移到 device')
x = torch.rand(1, 3, 512, 512) * 2 - 1
h = x.shape[2]
print(f'{"call":>4} {"interp":>8} {"venc":>9} {"vae_enc":>9} {"diT":>9} {"vae_dec":>9} {"total":>9}')
for i in range(6):
    t_all = time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        t_i, raw = timed(lambda: torch.nn.functional.interpolate(
            (0.5 * x + 0.5) * 255.0, size=m.dinov2_size, mode='bicubic').clamp(0., 255.))
        xn = (raw / 255.0 - mean) / std

        def venc():
            v = m.venc.prepare_tokens_with_masks(xn, None)
            zf = None
            for j, blk in enumerate(m.venc.blocks):
                v = blk(v)
                if j in m.layer_list:
                    zf = v[:, 1:].detach()
            return zf
        t_v, zfeat = timed(venc)

        def vae_enc():
            return m.vae.encode(x)
        t_ve, latd = timed(vae_enc)
        lat = latd.latent_dist.sample() * m.scale if hasattr(latd, 'latent_dist') else latd.mode()
        z = torch.randn(1, 4, h // 8, h // 8, device=obj.device, dtype=torch.float16)
        t_d, u = timed(lambda: m.diT(torch.cat([lat, z], 1),
                                     torch.ones(1, device=x.device),
                                     torch.zeros(1, device=x.device), [zfeat]))
        z1 = z - u
        t_vd, sr = timed(lambda: m.vae.decode(z1 / m.lat_std_inv + m.lat_mean))
    torch.cuda.synchronize()
    print(f'{i:>4} {t_i:8.0f} {t_v:9.0f} {t_ve:9.0f} {t_d:9.0f} {t_vd:9.0f} '
          f'{(time.time()-t_all)*1000:9.0f}', flush=True)
