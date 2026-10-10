# -*- coding: utf-8 -*-
"""用未 traced 的 UNet2DModel 逐块定位 CUDA graph 失败点"""
import os, sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch

torch.backends.cudnn.benchmark = False
d = torch.load(r'D:/models/tiny_v1/v2_s36000.pt', map_location='cpu', weights_only=False)
from diffusers import UNet2DModel
ch = tuple(d['ch'])
net = UNet2DModel(sample_size=256, in_channels=3, out_channels=3, layers_per_block=1,
                  block_out_channels=ch, down_block_types=('DownBlock2D',) * len(ch),
                  up_block_types=('UpBlock2D',) * len(ch))
net.load_state_dict(d['unet']); net = net.cuda().half().eval()
x = torch.randn(1, 3, 256, 256, device='cuda', dtype=torch.float16)

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
        os._exit(0)

with torch.no_grad():
    t = torch.tensor([500], dtype=torch.long, device='cuda')
    cap(lambda: net.conv_in(x), 'conv_in')
    cap(lambda: net.time_embedding(net.time_proj(t.float())), 'time_embed')
    h = net.conv_in(x)
    cap(lambda: net.down_blocks[0](h, net.time_embedding(net.time_proj(t.float()))), 'down[0]')
    cap(lambda: net.down_blocks[1](net.down_blocks[0](h, None)[0], None), 'down[0..1]')
    if not cap(lambda: net(x, t), 'FULL forward'):
        pass
print('LOCATE2_DONE')
