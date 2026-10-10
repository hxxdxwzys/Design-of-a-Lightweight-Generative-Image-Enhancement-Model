# -*- coding: utf-8 -*-
"""Q1 舍入规则精测 + Q4 差异定位。"""
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

# Q1: [3,1,8] -> 每通道一行，值取 k/255 与 (k+0.5)/255 边界
ks = [0, 1, 2, 3, 100, 200, 254, 255]
exact = torch.tensor([[[k / 255.0 for k in ks]]], dtype=torch.float32).repeat(3, 1, 1)
half = torch.tensor([[[(k + 0.5) / 255.0 for k in ks]]], dtype=torch.float32).repeat(3, 1, 1)
for name, t in (('exact', exact), ('half', half)):
    out = np.array(to_pil_image(t.contiguous()))
    print(name, 'pil=', out[0].tolist())
print('round_half_up(exact)=', torch.clamp(exact * 255 + 0.5, 0, 255).to(torch.uint8)[0, 0].tolist())
print('trunc(exact)       =', torch.clamp(exact * 255, 0, 255).to(torch.uint8)[0, 0].tolist())
print('round_half_up(half)=', torch.clamp(half * 255 + 0.5, 0, 255).to(torch.uint8)[0, 0].tolist())
print('trunc(half)        =', torch.clamp(half * 255, 0, 255).to(torch.uint8)[0, 0].tolist())
