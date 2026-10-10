# -*- coding: utf-8 -*-
"""试三种修法: (a) 多 warmup (b) 关 profiling executor (c) 查 attention 数量"""
import sys, torch
MD = r'D:/models/pkg_TinyV2g/model_dir'

def try_cap(m, x, n_warm):
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warm):
                m(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m(x)
        return g, None
    except Exception as e:
        return None, str(e)[:80]

m = torch.jit.load(MD + '/your_model.pt', map_location='cuda').eval()
x = torch.zeros(1, 3, 256, 256, device='cuda', dtype=torch.float16)
print('(a) 20 次 warmup 后 capture:', end=' ', flush=True)
g, err = try_cap(m, x, 20)
print('OK' if g else 'FAIL ' + err, flush=True)
if g is None:
    print('    -> 需要换进程重试 (上下文已污染)')
print('DONE_A')
