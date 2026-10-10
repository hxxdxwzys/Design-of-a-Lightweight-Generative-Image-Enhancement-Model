# -*- coding: utf-8 -*-
"""模型内部组件计时 v2 (2026-09-11): z = 随机噪声潜码 (与 runner.infer 一致)
用法: python profile_runner.py <model_dir> [次数]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
os.chdir(r'C:/models/finals')

md = Path(sys.argv[1] if len(sys.argv) > 1 else r'D:/models/final_compliant/model_dir')
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
sys.path.insert(0, str(md))
import runner as R  # noqa: E402

obj = R.Runner(str(md))
m = obj.model
x = torch.rand(1, 3, 512, 512) * 2 - 1


def t(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    r = fn()
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000, r


print(f'组件: venc/vae/diT ; dinov2_size={m.dinov2_size} layer_list={m.layer_list}', flush=True)
hdr = f'{"call":>4} {"interp":>8} {"venc":>9} {"vae_enc":>9} {"diT":>9} {"vae_dec":>9} {"total":>9}'
print(hdr, flush=True)
for i in range(N):
    t_all0 = time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        t_i, raw = t(lambda: F.interpolate((0.5 * x + 0.5) * 255.0,
                                           size=m.dinov2_size, mode='bicubic').clamp(0., 255.))
        xn = (raw / 255.0 - m.mean) / m.std

        def run_venc():
            v = m.venc.prepare_tokens_with_masks(xn, None)
            zf = None
            for j, blk in enumerate(m.venc.blocks):
                v = blk(v)
                if j in m.layer_list:
                    zf = v[:, 1:].detach()
            return zf
        t_v, zfeat = t(run_venc)
        t_ve, lat = t(lambda: m.vae.encode(x).mode())
        latn = (lat - m.lat_mean) * m.lat_std_inv
        z = torch.randn(1, 4, x.shape[2] // 8, x.shape[3] // 8,
                        device=obj.device, dtype=torch.float16)
        t_d, u = t(lambda: m.diT(torch.cat([latn, z], 1),
                                 torch.ones(1, device=x.device),
                                 torch.zeros(1, device=x.device), [zfeat]))
        z1 = z - u
        t_vd, sr = t(lambda: m.vae.decode(z1 / m.lat_std_inv + m.lat_mean))
    torch.cuda.synchronize()
    tot = (time.time() - t_all0) * 1000
    print(f'{i:>4} {t_i:8.0f} {t_v:9.0f} {t_ve:9.0f} {t_d:9.0f} {t_vd:9.0f} {tot:9.0f}', flush=True)
