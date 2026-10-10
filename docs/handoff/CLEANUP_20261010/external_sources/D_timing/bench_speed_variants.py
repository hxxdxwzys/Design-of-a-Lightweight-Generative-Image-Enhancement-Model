# -*- coding: utf-8 -*-
"""空闲 GPU 测速基准: v1(全512图) / v2(CPU切片) / v2b(GPU切片) — 官方方法论 (CPU输入+10次warmup+300次计时)"""
import sys
import time
import torch

sys.path.insert(0, r'D:\models\final_graph\model_dir')
sys.path.insert(0, r'D:\models\final_graph2\model_dir')
sys.path.insert(0, r'D:\models\final_graph2b\model_dir')

BASELINE = 287.0


def bench(runner, name):
    x = torch.randn(1, 3, 512, 512, dtype=torch.float32).clamp(-1, 1)
    xg = x.cuda()
    with torch.no_grad():
        for _ in range(10):
            runner.infer(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 300
        for _ in range(N):
            runner.infer(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N * 1000
        # GPU 输入
        for _ in range(10):
            runner.infer(xg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            runner.infer(xg)
        torch.cuda.synchronize()
        dt2 = (time.perf_counter() - t0) / N * 1000
    print(f'[{name}] CPU输入: {dt:.3f}ms -> {BASELINE/dt:.0f}x | GPU输入: {dt2:.3f}ms -> {BASELINE/dt2:.0f}x', flush=True)
    return dt, dt2


if __name__ == '__main__':
    from runner import Runner as R1
    import importlib
    sys.path.insert(0, r'D:\models\final_graph\model_dir')
    r1 = R1(r'D:\models\final_graph\model_dir')
    bench(r1, 'v1 全512图')

    sys.path.insert(0, r'D:\models\final_graph2\model_dir')
    from runner import Runner as R2
    r2 = R2(r'D:\models\final_graph2\model_dir')
    bench(r2, 'v2 CPU切片')

    sys.path.insert(0, r'D:\models\final_graph2b\model_dir')
    from runner import Runner as R2B
    r2b = R2B(r'D:\models\final_graph2b\model_dir')
    bench(r2b, 'v2b GPU切片')
