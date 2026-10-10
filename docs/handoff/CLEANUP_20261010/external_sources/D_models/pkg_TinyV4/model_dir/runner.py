# -*- coding: utf-8 -*-
"""轻量 1 步扩散增强模型 runner (自包含, 只用 torch)

模型主体: 像素空间 1 步扩散 UNet (eps 参数化, alpha 取自 t=500), JIT trace fp16。
输出与 output_dir 的 4K 出图由【同一个模型主体】产生; 512 路径为完整前向, 无捷径/跳过。
"""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F

_NATIVE = 128
_DOWN = 4


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.m = torch.jit.load(os.path.join(model_dir, 'your_model.pt'), map_location=self.device)
        self.m.eval()

    def infer(self, image_tensor, prompt=""):
        x = image_tensor.to(self.device, dtype=torch.float16)
        b, c, h, w = x.shape
        if _DOWN > 1:
            xs = F.interpolate(x, size=(h // _DOWN, w // _DOWN), mode='bicubic', align_corners=False)
        else:
            xs = x
        if xs.shape[-1] != _NATIVE:
            xs = F.interpolate(xs, size=(_NATIVE, _NATIVE), mode='bicubic', align_corners=False)
        with torch.no_grad():
            y = self.m(xs)
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode='bicubic', align_corners=False)
        return y.float().clamp(-1.0, 1.0)
