# -*- coding: utf-8 -*-
"""bench pkg_TinyV2g 的 graph runner (官方调用方式)"""
import sys, time, torch
MD = r'D:/models/pkg_TinyV2g/model_dir'
sys.path.insert(0, MD)
import runner as R
t0 = time.time(); r = R.Runner(MD)
print(f'load+capture {time.time()-t0:.2f}s | graph = {r.graph is not None}', flush=True)
x = (torch.rand(1, 3, 512, 512) * 2 - 1)
with torch.no_grad():
    y = r.infer(x)
    for _ in range(20):
        y = r.infer(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(200):
        y = r.infer(x)
    torch.cuda.synchronize()
ms = (time.perf_counter() - t) / 200 * 1000
print(f'[graph runner] 512: {ms:.2f} ms -> {285.6/ms:.1f}x | out {tuple(y.shape)} std={y.std().item():.3f} nan={bool(torch.isnan(y).any())}', flush=True)
print('BENCH_G_DONE')
