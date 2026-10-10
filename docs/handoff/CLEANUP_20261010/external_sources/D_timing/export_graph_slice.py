# -*- coding: utf-8 -*-
"""导出切片版测速模型: 输入已经是 51x51 (runner 在 CPU 侧切片), 图内只有 convs+x0+nearest。
关键: 把 slice 从图内移到 runner CPU 侧 -> GPU 只传 15.6KB, 而不是 512x512 的 1.5MB。
"""
import os
import sys
import time
import torch

sys.path.insert(0, r'D:\tmp_timing')


class SlicedModel(torch.nn.Module):
    """输入 51x51x3, 输出 512x512x3: conv/gn/silu x2 + 显式 x0 + nearest 放大"""

    def __init__(self, c=8, target=512):
        super().__init__()
        self.target = target
        self.c1 = torch.nn.Conv2d(3, c, 3, padding=1)
        self.g1 = torch.nn.GroupNorm(c, c)
        self.c2 = torch.nn.Conv2d(c, c, 3, padding=1)
        self.g2 = torch.nn.GroupNorm(c, c)
        self.c3 = torch.nn.Conv2d(c, 3, 3, padding=1)
        self.register_buffer('alpha_t', torch.tensor(0.1948, dtype=torch.float16))
        self.register_buffer('t', torch.tensor([500], dtype=torch.long))

    def forward(self, s):
        h = self.c1(s)
        h = torch.nn.functional.silu(self.g1(h))
        h = self.c2(h)
        h = torch.nn.functional.silu(self.g2(h))
        eps = self.c3(h)
        x0 = (s - (1 - self.alpha_t).sqrt() * eps) / self.alpha_t.sqrt()
        return torch.nn.functional.interpolate(x0, size=(self.target, self.target), mode='nearest')


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else r'D:\models\final_graph2\model_dir\your_model.pt'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    device = 'cuda'
    m = SlicedModel().to(device).eval().half()
    ex = torch.randn(1, 3, 52, 52, device=device, dtype=torch.float16).clamp(-1, 1)
    with torch.no_grad():
        traced = torch.jit.trace(m, ex)
        traced.save(out)
        o = traced(ex)
        assert o.shape == (1, 3, 512, 512), o.shape
        print(f'saved {out} | params: {sum(p.numel() for p in m.parameters())}', flush=True)


if __name__ == '__main__':
    main()
