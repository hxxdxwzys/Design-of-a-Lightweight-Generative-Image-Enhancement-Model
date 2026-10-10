# -*- coding: utf-8 -*-
"""导出 v4 测速模型 (v4b: 3->3 conv + silu + 3->3 conv + 显式 x0, 无 gn) + 动态尺寸检查"""
import sys
import os
import time
import torch

sys.path.insert(0, r'D:\tmp_timing')
from bench_v4 import FusedV4b  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else r'D:\models\final_graph4\model_dir\your_model.pt'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    device = 'cuda'
    m = FusedV4b().to(device).eval().half()
    ex = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float16).clamp(-1, 1)
    with torch.no_grad():
        traced = torch.jit.trace(m, ex)
        traced.save(out)
        ex256 = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float16)
        o256 = traced(ex256)
        assert o256.shape == (1, 3, 256, 256), f'dynamic size failed: {o256.shape}'
        print('dynamic-size check: 256 ->', tuple(o256.shape), flush=True)
        for _ in range(10):
            traced(ex)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            traced(ex)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 200 * 1000
    print(f'saved {out} | eager 512: {dt:.4f}ms', flush=True)
    print(f'params: {sum(p.numel() for p in m.parameters())}', flush=True)


if __name__ == '__main__':
    main()
