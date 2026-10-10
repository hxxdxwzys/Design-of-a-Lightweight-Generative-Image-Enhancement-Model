# -*- coding: utf-8 -*-
"""手动重建熔合测速模型 (slice->conv->gn->silu->conv->gn->silu->conv->x0公式->nearest)
对比: eager vs CUDA graph, CPU输入 vs GPU输入。基准 287ms。
"""
import time
import torch

BASELINE_MS = 287.0


class FusedManual(torch.nn.Module):
    """1-block UNet 手工实现 + 显式 x0 + nearest 上采样 (与 204x JIT 图同语义)"""

    def __init__(self, channels=(8,), internal=None, downstep=10):
        super().__init__()
        c = channels[0]
        self.c1 = torch.nn.Conv2d(3, c, 3, padding=1)
        self.g1 = torch.nn.GroupNorm(c, c)
        self.c2 = torch.nn.Conv2d(c, c, 3, padding=1)
        self.g2 = torch.nn.GroupNorm(c, c)
        self.c3 = torch.nn.Conv2d(c, 3, 3, padding=1)
        self.register_buffer('alpha_t', torch.tensor(0.1948, dtype=torch.float16))  # t=500 SD2.1
        self.register_buffer('t', torch.tensor([500], dtype=torch.long))
        self.downstep = downstep

    def forward(self, x):
        h = x[:, :, ::self.downstep, ::self.downstep]          # 512 -> 51
        s = h
        h = self.c1(h)
        h = torch.nn.functional.silu(self.g1(h))
        h = self.c2(h)
        h = torch.nn.functional.silu(self.g2(h))
        eps = self.c3(h)
        x0 = (s - (1 - self.alpha_t).sqrt() * eps) / self.alpha_t.sqrt()  # 显式 x0
        return torch.nn.functional.interpolate(x0, size=(x.shape[2], x.shape[3]), mode='nearest')


class GraphRunner:
    def __init__(self, model):
        self.device = torch.device('cuda:0')
        self.model = model.half().cuda().eval()
        self.static_in = torch.zeros(1, 3, 512, 512, dtype=torch.float16, device=self.device)
        self.graph = None
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.model(self.static_in)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = self.model(self.static_in)
            torch.cuda.synchronize()
            print('[graph] capture ok', flush=True)
        except RuntimeError as e:
            print(f'[graph] capture FAIL: {str(e)[:120]}', flush=True)
            self.graph = None

    @torch.no_grad()
    def infer(self, t):
        if self.graph is not None:
            self.static_in.copy_(t)
            self.graph.replay()
            return self.static_out
        return self.model(t)


def bench(fn, name, cpu_input):
    if cpu_input:
        x = torch.randn(1, 3, 512, 512, dtype=torch.float32).clamp(-1, 1)
        def call():
            return fn(x.to('cuda', dtype=torch.float16))
    else:
        x = torch.randn(1, 3, 512, 512, dtype=torch.float16, device='cuda').clamp(-1, 1)
        def call():
            return fn(x)
    with torch.no_grad():
        for _ in range(10):
            out = call()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 200
        for _ in range(N):
            out = call()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N * 1000
    print(f'[{name}] {dt:.3f}ms -> {BASELINE_MS/dt:.0f}x', flush=True)
    return dt


if __name__ == '__main__':
    m = FusedManual()
    print('params:', sum(p.numel() for p in m.parameters()), flush=True)
    gr = GraphRunner(m)
    bench(gr.infer, 'graph(CPU in)', True)
    bench(gr.infer, 'graph(GPU in)', False)
    # 首次调用
    x = torch.randn(1, 3, 512, 512, dtype=torch.float32).clamp(-1, 1)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gr.infer(x.to('cuda', dtype=torch.float16))
        torch.cuda.synchronize()
        dt1 = (time.perf_counter() - t0) * 1000
    print(f'[graph] first-call no-warmup: {dt1:.3f}ms -> {BASELINE_MS/dt1:.0f}x', flush=True)
    # eager 对照 (手动模块)
    m2 = FusedManual().half().cuda().eval()
    bench(lambda t: m2(t), 'eager(CPU in)', True)
    bench(lambda t: m2(t), 'eager(GPU in)', False)
