# -*- coding: utf-8 -*-
"""独立时延测量 v2 (2026-09-15 修: 输入必须上卡!)

v1 的坑: x = torch.rand(1,3,512,512) 留在 CPU -> 对带 torch.compile/dynamo 回退的 runner
(如 1.4B wave2), 每次调用都重走编译/搬运路径 => 把 0.98 s 测成 12.64 s (12 倍高估)。
v2: 输入 .to(cuda); warmup 30 次; 逐次 sync 计时 200 次; 报 median / mean / min。
"""
import sys, time, torch, statistics
md = sys.argv[1]
sys.path.insert(0, md)
import runner as R
dev = 'cuda'
r = R.Runner(md)
x = ((torch.rand(1, 3, 512, 512) * 2 - 1)).to(dev)
with torch.no_grad():
    for _ in range(30):
        y = r.infer(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(200):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = r.infer(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
med = statistics.median(ts)
print('[latency] median %.2f ms -> %.1fx | mean %.2f | min %.2f | graph=%s | std=%.3f nan=%s' % (
    med, 285.6 / med, sum(ts) / len(ts), min(ts),
    getattr(r, 'graph', None) is not None, y.std().item(), bool(torch.isnan(y).any())))
print('LATENCY_DONE')
