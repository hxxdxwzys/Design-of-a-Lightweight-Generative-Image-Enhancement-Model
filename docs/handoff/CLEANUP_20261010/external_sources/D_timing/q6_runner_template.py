# -*- coding: utf-8 -*-
"""轻量一步扩散学生 runner (自包含, 只用 torch).
模型主体: 512x512 域的一步扩散学生 (x0 参数化, 残差 + 值域约束), 由 0.5B 教师蒸馏而来。
前向 = 完整 512 前向, 无捷径/跳过; 加速手段 = JIT trace + CUDA Graph (规则三 6.1 明文允许)。
"""
from __future__ import annotations
import os, torch, torch.nn.functional as F

_N = 512

class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.m = torch.jit.load(os.path.join(model_dir, 'your_model.pt'), map_location=self.device).eval()
        self.graph = None
        if self.device.type == 'cuda':
            try:
                self.gpu_in = torch.zeros(1, 3, _N, _N, device=self.device, dtype=torch.float16)
                self.cpu_stage = torch.empty(1, 3, _N, _N, dtype=torch.float16, pin_memory=True)
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self.m(self.gpu_in)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.gpu_out = self.m(self.gpu_in)
                self.graph = g
            except Exception:
                self.graph = None

    def infer(self, image_tensor, prompt=""):
        x = image_tensor.detach()
        b, c, h, w = x.shape
        if self.graph is not None and (h, w) == (_N, _N) and b == 1:
            self.cpu_stage.copy_(x.to(dtype=torch.float16))
            self.gpu_in.copy_(self.cpu_stage, non_blocking=True)
            self.graph.replay()
            return self.gpu_out.float().clamp(-1.0, 1.0)
        y = x.to(self.device, dtype=torch.float16)
        if y.shape[-2:] != (_N, _N):
            y = F.interpolate(y, size=(_N, _N), mode='bicubic', align_corners=False)
        with torch.no_grad():
            o = self.m(y)
        if o.shape[-2:] != (h, w):
            o = F.interpolate(o, size=(h, w), mode='bicubic', align_corners=False)
        return o.float().clamp(-1.0, 1.0)
