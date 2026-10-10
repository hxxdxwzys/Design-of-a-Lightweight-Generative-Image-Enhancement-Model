# -*- coding: utf-8 -*-
"""CUDA Graph 提速验证 (2026-09-14): 微模型 14.6ms 的地板是内核启动开销, 不是 FLOPs
做法: 把 traced 模块 capture 成 CUDA graph, replay 计时 + 与 eager 输出比对
"""
import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

md = Path(r'D:/models/pkg_TinyV2/model_dir')
sys.path.insert(0, str(md))
import importlib.util
spec = importlib.util.spec_from_file_location('t2runner', str(md / 'runner.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
R = mod.Runner(str(md))
net = R.m.eval()

def bench(fn, n=200, warm=20):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000

# eager runner 路径
x512 = (torch.rand(1, 3, 512, 512) * 2 - 1)
ms_eager = bench(lambda: R.infer(x512))
print(f'[eager runner] 512: {ms_eager:.2f} ms -> {285.6/ms_eager:.1f}x', flush=True)

# 只看模型本身 (256 内部)
x256 = F.interpolate(x512.to('cuda').half(), size=(256, 256), mode='bicubic', align_corners=False).contiguous()
ms_model = bench(lambda: net(x256))
print(f'[eager model] 256: {ms_model:.2f} ms -> {285.6/ms_model:.1f}x', flush=True)

# 纯 GPU kernel 时间 (用 graph 之前先看同步开销)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(200):
    net(x256)
torch.cuda.synchronize()
print(f'[eager, 无逐次同步] 200 次总耗时 {(time.perf_counter()-t0):.3f}s -> {(time.perf_counter()-t0)/200*1000:.2f} ms/次', flush=True)

# ---- CUDA Graph ----
try:
    static_in = x256.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            net(static_in)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            static_out = net(static_in)
    # 验证一致性
    ref = net(x256.clone())
    static_in.copy_(x256)
    g.replay(); torch.cuda.synchronize()
    d = (ref - static_out).abs().max().item()
    ms_graph = bench(lambda: g.replay())
    print(f'[CUDA graph] replay: {ms_graph:.2f} ms -> {285.6/ms_graph:.1f}x  (与 eager 最大差 {d:.5f})', flush=True)

    # 端到端 (含 512->256 缩放 + 拷入拷出)
    def full_graph():
        xs = F.interpolate(x512.to('cuda').half(), size=(256, 256), mode='bicubic', align_corners=False)
        static_in.copy_(xs)
        g.replay()
        return F.interpolate(static_out, size=(512, 512), mode='bicubic', align_corners=False)
    ms_full = bench(full_graph, n=100, warm=20)
    print(f'[CUDA graph 端到端 512] {ms_full:.2f} ms -> {285.6/ms_full:.1f}x', flush=True)
except Exception as e:
    print(f'[CUDA graph] 失败: {type(e).__name__}: {str(e)[:200]}')
print('GRAPH_TEST_DONE')
