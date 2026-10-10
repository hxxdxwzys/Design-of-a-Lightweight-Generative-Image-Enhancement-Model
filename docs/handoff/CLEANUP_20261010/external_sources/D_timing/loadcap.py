# -*- coding: utf-8 -*-
"""定位: 加载版 traced 模块为何 capture 失败 (打印真实异常)"""
import os, sys, traceback
import torch

MD = r'D:/models/pkg_TinyV2g/model_dir'
m = torch.jit.load(MD + '/your_model.pt', map_location='cuda')
m.eval()
x = torch.zeros(1, 3, 256, 256, device='cuda', dtype=torch.float16)
print('loaded, out', tuple(m(x).shape), flush=True)

try:
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            m(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = m(x)
    print('capture OK', flush=True)
except Exception as e:
    print('capture FAILED:', type(e).__name__, str(e)[:300], flush=True)
    traceback.print_exc()
print('LOADCAP_DONE')
