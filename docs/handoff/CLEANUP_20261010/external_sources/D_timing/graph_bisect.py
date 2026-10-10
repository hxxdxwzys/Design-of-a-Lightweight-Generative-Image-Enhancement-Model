# -*- coding: utf-8 -*-
import torch
from diffusers import UNet2DModel

def try_capture(name, fn, x):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(x)
        torch.cuda.synchronize()
        print(f'{name}: CAPTURE OK', flush=True)
    except RuntimeError as e:
        print(f'{name}: FAIL {str(e)[:150]}', flush=True)

x = torch.zeros(1, 3, 512, 512, dtype=torch.float16, device='cuda')

unet = UNet2DModel(sample_size=512, in_channels=3, out_channels=3, layers_per_block=1,
                   block_out_channels=(8,), down_block_types=('DownBlock2D',) * 1,
                   up_block_types=('UpBlock2D',) * 1).half().cuda().eval()
try_capture('unet2d', lambda t: unet(t, torch.tensor([500], device='cuda')), x)

a = torch.tensor(0.5, dtype=torch.float16, device='cuda')
try_capture('formula', lambda t: (t - (1 - a).sqrt() * t) / a.sqrt(), x)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 8, 3, padding=1); self.g1 = torch.nn.GroupNorm(8, 8)
        self.c2 = torch.nn.Conv2d(8, 8, 3, padding=1); self.g2 = torch.nn.GroupNorm(8, 8)
        self.c3 = torch.nn.Conv2d(8, 3, 3, padding=1)

    def forward(self, t):
        h = t[:, :, ::10, ::10]
        h = self.c1(h); h = torch.nn.functional.silu(self.g1(h))
        h = self.c2(h); h = torch.nn.functional.silu(self.g2(h))
        h = self.c3(h)
        return torch.nn.functional.interpolate(h, size=(512, 512), mode='nearest')


m = M().half().cuda().eval()
try_capture('manual-module', m, x)
print('all done', flush=True)
