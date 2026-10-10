# -*- coding: utf-8 -*-
"""出货版 runner 的组件分解 (2026-09-13) —— 用 runner 自己的模块, 不是 diffusers 版
前置: speed_opt_probe 已证 cudnn/jit/channels_last 均无效(366/386/444ms) => eager 已近算力上限,
      要提速只能【减少计算量】。本脚本回答: 减哪一块最有效。
用法: python speed_breakdown_shipped.py [--n 20]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MD = Path(r'D:/models/opt_probe/model_dir')
BASELINE_MS = 285.6
OUT = Path(r'D:/tmp_timing/speed_breakdown_shipped.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    a = ap.parse_args()
    import torch

    sys.path.insert(0, str(MD)); sys.path.insert(0, str(MD / '_vendor'))
    import runner as R
    import torch.nn.functional as F

    r = R.Runner(str(MD))
    m = r.model
    x = (torch.rand(1, 3, 512, 512, device='cuda') * 2 - 1)
    z = torch.randn(1, 4, 64, 64, device='cuda', dtype=torch.float16)

    def timeit(fn, n=None, warm=5, label=''):
        n = n or a.n
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        med = ts[len(ts) // 2]
        print(f'  {label:<34} 中位 {med:8.1f} ms', flush=True)
        return med

    print('\n=== 出货版组件分解 ===', flush=True)
    full = timeit(lambda: r.infer(x), label='全链 infer(512)')

    xh = x.half()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        raw = F.interpolate((0.5 * xh + 0.5) * 255.0, size=m.dinov2_size, mode='bicubic').clamp(0., 255.)
        xn = (raw / 255.0 - m.mean) / m.std
        v = m.venc.prepare_tokens_with_masks(xn, None)
        zfeat = None
        for i, blk in enumerate(m.venc.blocks):
            v = blk(v)
            if i in m.layer_list:
                zfeat = v[:, 1:].detach()
        lat = m.vae.encode(xh).latent_dist.sample() * m.scale
        u = m.diT(torch.cat([lat, z], 1), torch.ones(1, device='cuda'),
                  torch.zeros(1, device='cuda'), [zfeat])
        z1 = (z - u) / m.scale

    def dino():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            raw = F.interpolate((0.5 * xh + 0.5) * 255.0, size=m.dinov2_size, mode='bicubic').clamp(0., 255.)
            xn = (raw / 255.0 - m.mean) / m.std
            v = m.venc.prepare_tokens_with_masks(xn, None)
            for i, blk in enumerate(m.venc.blocks):
                v = blk(v)
                if i in m.layer_list:
                    return v[:, 1:].detach()

    def vae_enc():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            return m.vae.encode(xh).latent_dist.sample() * m.scale

    def dit():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            return m.diT(torch.cat([lat, z], 1), torch.ones(1, device='cuda'),
                         torch.zeros(1, device='cuda'), [zfeat])

    def dec():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            return m.dec(z1).clamp(-1, 1)

    parts = {}
    parts['dino_l8'] = timeit(dino, label='DINOv2 (到第8层, 448)')
    parts['vae_encode'] = timeit(vae_enc, label='VAE encode (shim)')
    parts['dit'] = timeit(dit, label=f'DiT 1步 ({len(m.diT.blocks)} blocks)')
    parts['decoder'] = timeit(dec, label='LightDecoder decode')

    # DiT 深度截断 (直接改 blocks 列表, 实测)
    dit_blocks = list(m.diT.blocks)
    prune = {}
    for keep in (14, 7, 4):
        m.diT.blocks = torch.nn.ModuleList(dit_blocks[:keep])
        ms = timeit(dit, n=max(8, a.n // 2), warm=2, label=f'DiT {keep} blocks')
        prune[keep] = ms
    m.diT.blocks = torch.nn.ModuleList(dit_blocks)

    s = sum(parts.values())
    print(f'\n  组件和 {s:.1f} ms vs 全链 {full:.1f} ms (差 {full-s:+.1f})')
    print(f'  当前比基线 {BASELINE_MS/full:.2f}x')
    print('\n  剪枝后全链估计:')
    est = {}
    for keep, ms in prune.items():
        tot = full - parts['dit'] + ms
        est[keep] = {'dit_ms': round(ms, 1), 'full_est_ms': round(tot, 1),
                     'ratio': round(BASELINE_MS / tot, 2)}
        print(f'    DiT {keep:2d}/28 -> 全链 ~{tot:.0f} ms -> 比基线 {BASELINE_MS/tot:.2f}x')

    OUT.write_text(json.dumps({'full_ms': round(full, 1), 'baseline_ms': BASELINE_MS,
                               'ratio_now': round(BASELINE_MS / full, 3),
                               'parts_ms': {k: round(v, 1) for k, v in parts.items()},
                               'share': {k: round(v / s * 100, 1) for k, v in parts.items()},
                               'dit_prune': est}, indent=1), encoding='utf-8')
    print(f'\n已存 {OUT}')
    print('BREAKDOWN_SHIPPED_DONE')


if __name__ == '__main__':
    main()
