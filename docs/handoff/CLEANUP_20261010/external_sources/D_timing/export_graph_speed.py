# -*- coding: utf-8 -*-
"""导出测速模型 your_model.pt (熔合 1-block UNet + 显式 x0, 动态尺寸 trace)
runner.py 内做 CUDA Graph 捕获/重放; 此脚本只导出 JIT 图。
"""
import sys
import time
import torch

sys.path.insert(0, r'D:\tmp_timing')
from graph_speed import FusedManual


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else r'D:\models\final_graph\model_dir\your_model.pt'
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    device = 'cuda'
    m = FusedManual().to(device).eval().half()
    ex = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float16).clamp(-1, 1)
    with torch.no_grad():
        traced = torch.jit.trace(m, ex)
        traced.save(out)
        # 动态尺寸验证: 256 输入应得 256 输出
        ex256 = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float16)
        o256 = traced(ex256)
        assert o256.shape == (1, 3, 256, 256), f'dynamic size failed: {o256.shape}'
        print('dynamic-size check: 256 ->', tuple(o256.shape), flush=True)
        # eager 测速 (会与训练争 GPU, 数字仅参考)
        for _ in range(5):
            traced(ex)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            traced(ex)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 100 * 1000
    print(f'saved {out} | eager 512: {dt:.3f}ms (~{287/dt:.0f}x, 与训练争卡)', flush=True)
    print(f'params: {sum(p.numel() for p in m.parameters())}', flush=True)


if __name__ == '__main__':
    main()
