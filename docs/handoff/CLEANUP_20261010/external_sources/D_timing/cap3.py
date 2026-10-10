# -*- coding: utf-8 -*-
"""验证: 去掉 x.shape[0] 依赖后, traced 模块能否 capture"""
import os, sys, time
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch, torch.nn as nn
torch.backends.cudnn.benchmark = False

class Infer2(nn.Module):
    def __init__(self, unet, alpha, t=500):
        super().__init__()
        self.unet = unet
        self.register_buffer('sq_a', torch.tensor(float(alpha) ** 0.5))
        self.register_buffer('sq_1a', torch.tensor(float(1 - alpha) ** 0.5))
        self.register_buffer('tt', torch.tensor([t], dtype=torch.long))   # 固定, 不依赖输入形状
    def forward(self, x):
        pred = self.unet(x, self.tt).sample
        return (x - self.sq_1a.to(x.dtype) * pred) / self.sq_a.to(x.dtype)

d = torch.load(r'D:/models/tiny_v1/v2_s36000.pt', map_location='cpu', weights_only=False)
from diffusers import UNet2DModel
ch = tuple(d['ch'])
unet = UNet2DModel(sample_size=256, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=ch, down_block_types=('DownBlock2D',) * len(ch),
                   up_block_types=('UpBlock2D',) * len(ch))
unet.load_state_dict(d['unet'])
net = Infer2(unet, float(d['alpha'])).cuda().eval().half()
x = torch.randn(1, 3, 256, 256, device='cuda', dtype=torch.float16)
with torch.no_grad():
    tr = torch.jit.trace(net, x)
    ref = tr(x)
print('[trace] ok, out', tuple(ref.shape), flush=True)

def cap(fn, warm=3):
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warm):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        return g, out, None
    except Exception as e:
        return None, None, f'{type(e).__name__}: {str(e)[:130]}'

g, out, err = cap(lambda: tr(x))
print('[capture]', 'OK' if g else 'FAIL ' + str(err), flush=True)
if g:
    g.replay(); torch.cuda.synchronize()
    print('  replay maxdiff =', float((ref - out).abs().max()), flush=True)
    with torch.no_grad():
        for _ in range(20): g.replay()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(300): g.replay()
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 300 * 1000
    with torch.no_grad():
        for _ in range(20): tr(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(300): tr(x)
        torch.cuda.synchronize()
    ms2 = (time.perf_counter() - t0) / 300 * 1000
    print(f'  graph replay {ms:.2f} ms -> {285.6/ms:.1f}x | traced eager {ms2:.2f} ms -> {285.6/ms2:.1f}x', flush=True)
print('CAP3_DONE')
