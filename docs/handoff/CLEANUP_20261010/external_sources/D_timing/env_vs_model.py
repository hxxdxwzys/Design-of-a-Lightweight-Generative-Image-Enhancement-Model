# -*- coding: utf-8 -*-
"""环境 vs 模型 的周期性尖峰对照 (2026-09-11)
A) 纯 GPU 计算循环(大 matmul)  B) 纯显存分配循环  C) 我们的 runner
若 A/B 也出现"每 N 次一次尖峰" => 环境问题(驱动/电源/分配器); 否则是模型内部
"""
from __future__ import annotations

import statistics
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch


def loop(name, fn, n=12):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000)
    spikes = [i for i, v in enumerate(ts) if v > 1000]
    print(f'{name}')
    print('   ' + ' '.join(f'{v:.0f}' for v in ts))
    print(f'   中位 {statistics.median(ts):.0f} ms | 均值 {statistics.mean(ts):.0f} | '
          f'尖峰 {len(spikes)}/{n} @ {spikes}')


print('=== 机器/环境对照 ===')
a = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
b = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
loop('A) 大 matmul 2048x2048 fp16', lambda: a @ b)

buf = []
loop('B) 每次新分配 256MB fp16', lambda: buf.append(torch.empty(128 * 1024 * 1024,
                                                               dtype=torch.half, device='cuda')))
del buf
torch.cuda.empty_cache()

# 换用 expandable_segments 看是否改变
print('\n（如 A/B 无尖峰而 C 有 → 问题在我们的模型）')
