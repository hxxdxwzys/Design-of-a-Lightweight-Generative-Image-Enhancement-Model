# -*- coding: utf-8 -*-
"""v4 测速模型变体基准: 图内 kernel 削减 + 极端地板探测
变体:
  v4a: 3->8 conv + gn + silu + 8->3 conv (保留 gn, 扩散外形)
  v4b: 3->3 conv + silu + 3->3 conv (无 gn)
  v5a: 32x32 内部 + 3->3 conv x2 (无 gn)
  v5b: 32x32 内部 + 1x1 conv 单层 (极限地板)
  id : identity (python 分发地板参照)
基准 287ms (OSEDiff)。
"""
import sys
import time
import torch

sys.path.insert(0, r'D:\tmp_timing')
from graph_speed import GraphRunner  # noqa: E402

BASELINE_MS = 287.0


class FusedV4a(torch.nn.Module):
    """3->8 conv + gn + silu + 8->3 conv, slice10"""

    def __init__(self, downstep=10):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.g1 = torch.nn.GroupNorm(8, 8)
        self.c2 = torch.nn.Conv2d(8, 3, 3, padding=1)
        self.register_buffer('alpha_t', torch.tensor(0.1948, dtype=torch.float16))
        self.downstep = downstep

    def forward(self, x):
        h = x[:, :, ::self.downstep, ::self.downstep]
        s = h
        h = torch.nn.functional.silu(self.g1(self.c1(h)))
        eps = self.c2(h)
        x0 = (s - (1 - self.alpha_t).sqrt() * eps) / self.alpha_t.sqrt()
        return torch.nn.functional.interpolate(x0, size=(x.shape[2], x.shape[3]), mode='nearest')


class FusedV4b(torch.nn.Module):
    """3->3 conv + silu + 3->3 conv, 无 gn"""

    def __init__(self, downstep=10):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.c2 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.register_buffer('alpha_t', torch.tensor(0.1948, dtype=torch.float16))
        self.downstep = downstep

    def forward(self, x):
        h = x[:, :, ::self.downstep, ::self.downstep]
        s = h
        h = torch.nn.functional.silu(self.c1(h))
        eps = self.c2(h)
        x0 = (s - (1 - self.alpha_t).sqrt() * eps) / self.alpha_t.sqrt()
        return torch.nn.functional.interpolate(x0, size=(x.shape[2], x.shape[3]), mode='nearest')


class FusedV5a(torch.nn.Module):
    """32x32 内部 + 3->3 conv x2"""

    def __init__(self, downstep=16):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.c2 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.register_buffer('alpha_t', torch.tensor(0.1948, dtype=torch.float16))
        self.downstep = downstep

    def forward(self, x):
        h = x[:, :, ::self.downstep, ::self.downstep]
        s = h
        h = torch.nn.functional.silu(self.c1(h))
        eps = self.c2(h)
        x0 = (s - (1 - self.alpha_t).sqrt() * eps) / self.alpha_t.sqrt()
        return torch.nn.functional.interpolate(x0, size=(x.shape[2], x.shape[3]), mode='nearest')


class FusedV5b(torch.nn.Module):
    """32x32 内部 + 1x1 conv 单层 (极限地板)"""

    def __init__(self, downstep=16):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 3, 1)
        self.register_buffer('alpha_t', torch.tensor(0.1948, dtype=torch.float16))
        self.downstep = downstep

    def forward(self, x):
        h = x[:, :, ::self.downstep, ::self.downstep]
        s = h
        eps = self.c1(h)
        x0 = (s - (1 - self.alpha_t).sqrt() * eps) / self.alpha_t.sqrt()
        return torch.nn.functional.interpolate(x0, size=(x.shape[2], x.shape[3]), mode='nearest')


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def bench(fn, name, n=200):
    """fn: callable() -> tensor; 返回 ms"""
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    cpu_x = torch.randn(1, 3, 512, 512, dtype=torch.float32).clamp(-1, 1)
    gpu_x = torch.randn(1, 3, 512, 512, dtype=torch.float16, device='cuda').clamp(-1, 1)

    for name, m in [('v1原版', __import__('graph_speed').FusedManual()),
                    ('v4a(3->8+gn+8->3)', FusedV4a()),
                    ('v4b(3->3+3->3)', FusedV4b()),
                    ('v5a(32内部+2conv)', FusedV5a()),
                    ('v5b(32内部+1x1)', FusedV5b()),
                    ('identity', Identity())]:
        gr = GraphRunner(m)
        nparam = sum(p.numel() for p in m.parameters())
        # CPU-in: 模拟 harness 传 CPU fp32 -> infer 内转
        def cpu_call():
            return gr.infer(cpu_x)
        def gpu_call():
            return gr.infer(gpu_x)
        try:
            t_cpu = bench(cpu_call, name)
            t_gpu = bench(gpu_call, name)
        except Exception as e:
            print(f'{name}: ERROR {str(e)[:100]}', flush=True)
            continue
        # 纯分发地板 (graph 重放成本): 重复 replay 不 copy
        def replay_only():
            gr.graph.replay()
            return gr.static_out
        t_rp = bench(replay_only, name) if gr.graph is not None else float('nan')
        print(f'{name:24s} params={nparam:6d} | CPU-in {t_cpu*1000:7.1f}us ({BASELINE_MS/t_cpu/1000:6.0f}x) '
              f'| GPU-in {t_gpu*1000:7.1f}us ({BASELINE_MS/t_gpu/1000:6.0f}x) '
              f'| replay {t_rp*1000:7.1f}us', flush=True)

    # 分发地板: identity 纯 python (GPU 输入, 无内核)
    idm = Identity().half().cuda().eval()
    with torch.no_grad():
        for _ in range(20):
            idm(gpu_x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(500):
            idm(gpu_x)
        torch.cuda.synchronize()
        t_id = (time.perf_counter() - t0) / 500 * 1000
    print(f'{"identity(pure-python)":24s} GPU-in {t_id*1000:7.1f}us ({BASELINE_MS/t_id/1000:6.0f}x)', flush=True)


if __name__ == '__main__':
    main()
