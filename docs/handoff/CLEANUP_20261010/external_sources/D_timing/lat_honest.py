# -*- coding: utf-8 -*-
"""诚实时延协议（含 GPU 同步）：
  预热 30 s -> 3 轮 x (100 次调用 + 一次同步) 均摊 -> 报 median/min/p90 of the 3 rounds。
  另报单次前后同步的分布，便于对照。
用法: python lat_honest.py <model_dir> [<model_dir> ...]
"""
import importlib.util, os, statistics, sys, time
import torch

dirs = sys.argv[1:]
runners, names = [], []
for d in dirs:
    spec = importlib.util.spec_from_file_location("r_%d" % len(runners), os.path.join(d, "runner.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    R = m.Runner(d)
    runners.append(R); names.append(os.path.basename(os.path.dirname(d)))

x = (torch.rand(1, 3, 512, 512, device="cuda") * 2 - 1).contiguous()
with torch.no_grad():
    t0 = time.time(); i = 0
    while time.time() - t0 < 30.0:
        runners[i % len(runners)].infer(x, prompt=""); i += 1
    torch.cuda.synchronize()
print("[preheat] %.0f s / %d iters" % (time.time() - t0, i), flush=True)

N = 100
amort = {n: [] for n in names}
for rnd in range(3):
    for n, R in zip(names, runners):
        with torch.no_grad():
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(N):
                R.infer(x, prompt="")
            torch.cuda.synchronize()
            amort[n].append((time.perf_counter() - t) * 1e3 / N)

print("\n%-42s %10s %10s %10s  (amortized over %d calls x 3 rounds, GPU-synced)" % ("model", "median", "min", "p90", N))
for n in names:
    s = sorted(amort[n])
    print("%-42s %10.2f %10.2f %10.2f" % (n, statistics.median(s), s[0], s[-1]))

print("\n-- single-call, sync before+after (30 calls) --")
for n, R in zip(names, runners):
    ts = []
    with torch.no_grad():
        for _ in range(30):
            torch.cuda.synchronize(); t = time.perf_counter()
            R.infer(x, prompt=""); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
    s = sorted(ts)
    print("%-42s median=%8.2f  min=%8.2f  p90=%8.2f" % (n, statistics.median(s), s[0], s[int(0.9*len(s))]))
