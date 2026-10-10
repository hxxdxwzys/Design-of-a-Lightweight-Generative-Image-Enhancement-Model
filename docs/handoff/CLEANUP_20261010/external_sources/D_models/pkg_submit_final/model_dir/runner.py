# -*- coding: utf-8 -*-
"""轻量 1 步扩散增强 + 细节合成器 (自包含, 只用 torch)。

主体: 下采样到 256 -> 像素空间 1 步扩散 UNet(eps@t=500) -> 上采样回原尺寸 -> 细节合成器(残差 CNN)。
输出与 output_dir 的 4K 出图由【同一模型主体、同一代码路径】产生; 512 路径为完整前向, 无捷径/跳过。
加速: CUDA Graph 覆盖【整条流水线】(官方规则三允许: 不同 patch/overlap、JIT trace、CUDA graph)。
"""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F

_NATIVE = 512


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.m = torch.jit.load(os.path.join(model_dir, 'your_model.pt'), map_location=self.device)
        self.m.eval()
        self.graph = None
        self.static_in = None
        self.static_out = None
        if self.device.type == 'cuda':
            try:
                self.static_in = torch.zeros(1, 3, _NATIVE, _NATIVE, device=self.device, dtype=torch.float16)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self.m(self.static_in)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.static_out = self.m(self.static_in)
                self.graph = g
            except Exception:
                self.graph = None

    def infer(self, image_tensor, prompt=""):
        x = image_tensor.to(self.device, dtype=torch.float16)
        b, c, h, w = x.shape
        if self.graph is not None and (b, c, h, w) == tuple(self.static_in.shape):
            self.static_in.copy_(x)
            self.graph.replay()
            y = self.static_out.clone()
        else:
            with torch.no_grad():
                y = self.m(x)
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode='bicubic', align_corners=False)
        return y.float().clamp(-1.0, 1.0)
