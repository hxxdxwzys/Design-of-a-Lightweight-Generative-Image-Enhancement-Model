# -*- coding: utf-8 -*-
"""定位 CUDA graph 失败的 op + 备选方案 (jit optimize / fp32)"""
import os, sys, time
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from pathlib import Path
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = False
md = Path(r'D:/models/pkg_TinyV2/model_dir')
sys.path.insert(0, str(md))
import importlib.util
spec = importlib.util.spec_from_file_location('t2runner', str(md / 'runner.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
R = mod.Runner(str(md))
net = R.m.eval()
x256 = torch.randn(1, 3, 256, 256, device='cuda', dtype=torch.float16)

def cap(fn, label):
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        print(f'  [OK] {label}', flush=True)
        return True
    except Exception as e:
        print(f'  [FAIL] {label}: {type(e).__name__}: {str(e)[:110]}', flush=True)
        os._exit(0)   # 上下文已污染, 直接退出

with torch.no_grad():
    # 逐块定位: 先进 embedding + down
    h = net.conv_in(x256)
    cap(lambda: net.conv_in(x256), 'conv_in')
    cap(lambda: (net.time_proj(torch.tensor([500.0], device='cuda')), net.time_embedding(net.time_proj(torch.tensor([500.0], device='cuda')))), 'time embed')
    cap(lambda: net.down_blocks[0](h, None), 'down_blocks[0]')
    cap(lambda: net.down_blocks[3](h, None), 'down_blocks[3]')
    cap(lambda: net.up_blocks[0](h, None, None), 'up_blocks[0]')
    cap(lambda: net.conv_out(h), 'conv_out')
    cap(lambda: net(x256), 'full forward')
print('LOCATE_DONE')
