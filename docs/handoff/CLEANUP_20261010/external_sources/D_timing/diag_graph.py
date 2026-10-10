# -*- coding: utf-8 -*-
"""CUDA Graph 诊断: (1) 简单 conv 能否 capture (2) 我们的模型 (3) jit optimize 备选"""
import sys, time
from pathlib import Path
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = False

def try_capture(fn, warm=5):
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warm):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g, None
    except Exception as e:
        return None, f'{type(e).__name__}: {str(e)[:130]}'

x = torch.randn(1, 32, 64, 64, device='cuda', dtype=torch.float16)
conv = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU()).cuda().half().eval()
g, err = try_capture(lambda: conv(x))
print('[1] simple conv+GN+SiLU:', 'OK capturable' if g else 'FAIL ' + str(err), flush=True)

md = Path(r'D:/models/pkg_TinyV2/model_dir')
sys.path.insert(0, str(md))
import importlib.util
spec = importlib.util.spec_from_file_location('t2runner', str(md / 'runner.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
R = mod.Runner(str(md))
net = R.m.eval()
x256 = torch.randn(1, 3, 256, 256, device='cuda', dtype=torch.float16)
g2, err2 = try_capture(lambda: net(x256))
print('[2] traced tiny unet:', 'OK capturable' if g2 else 'FAIL ' + str(err2), flush=True)

try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    opt = torch.jit.optimize_for_inference(net)
    with torch.no_grad():
        for _ in range(10):
            opt(x256)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(100):
            opt(x256)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / 100 * 1000
    ref = net(x256); o = opt(x256)
    print(f'[3] jit optimize_for_inference: {ms:.2f} ms -> {285.6/ms:.1f}x (maxdiff {(ref-o).abs().max().item():.5f})', flush=True)
except Exception as e:
    print('[3] jit optimize FAIL:', type(e).__name__, str(e)[:150], flush=True)
print('GRAPH_DIAG_DONE')
