# -*- coding: utf-8 -*-
"""① 未 traced 模型能否 capture (用固定 t) ② traced 版本为何不行 -> 定位 aten::size"""
import os, sys, time, traceback
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch

torch.backends.cudnn.benchmark = False

def cap(fn, label):
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        return g, out, None
    except Exception as e:
        return None, None, f'{type(e).__name__}: {str(e)[:150]}'

d = torch.load(r'D:/models/tiny_v1/v2_s36000.pt', map_location='cpu', weights_only=False)
from diffusers import UNet2DModel
ch = tuple(d['ch'])
net = UNet2DModel(sample_size=256, in_channels=3, out_channels=3, layers_per_block=1,
                  block_out_channels=ch, down_block_types=('DownBlock2D',) * len(ch),
                  up_block_types=('UpBlock2D',) * len(ch))
net.load_state_dict(d['unet']); net = net.cuda().half().eval()
x = torch.randn(1, 3, 256, 256, device='cuda', dtype=torch.float16)
t = torch.tensor([500], dtype=torch.long, device='cuda')

with torch.no_grad():
    g, out, err = cap(lambda: net(x, t).sample, 'untraced UNet (fixed t)')
    print('[1] untraced UNet capture:', 'OK' if g else 'FAIL ' + str(err), flush=True)
    if g:
        ref = net(x, t).sample
        g.replay(); torch.cuda.synchronize()
        print('    replay maxdiff =', float((ref - out).abs().max()), flush=True)
        with torch.no_grad():
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(200):
                g.replay()
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 200 * 1000
        print(f'    graph replay = {ms:.2f} ms -> {285.6/ms:.1f}x', flush=True)
        with torch.no_grad():
            for _ in range(20):
                net(x, t)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(200):
                net(x, t)
            torch.cuda.synchronize()
        ms2 = (time.perf_counter() - t0) / 200 * 1000
        print(f'    eager       = {ms2:.2f} ms -> {285.6/ms2:.1f}x', flush=True)
print('CAP2_DONE')
