# -*- coding: utf-8 -*-
"""严格同会话时延台（与 使用说明.md §7 协议一致）：预热 -> 3 轮 x 20 次 -> median/min。
用法: python lat_measure.py <model_dir> [<model_dir> ...]
"""
import importlib.util, os, statistics, sys, time
import torch

dirs = sys.argv[1:]
assert dirs, "need at least one model_dir"
runners, names = [], []
for d in dirs:
    spec = importlib.util.spec_from_file_location("r_%d" % len(runners), os.path.join(d, "runner.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    R = m.Runner(d)
    runners.append(R); names.append(os.path.basename(os.path.dirname(d)) + "/" + os.path.basename(d))
    print("[load] %-52s graph=%s device=%s" % (names[-1], R.graph is not None, R.device), flush=True)

x = (torch.rand(1, 3, 512, 512, device="cuda") * 2 - 1).contiguous()
PREHEAT_S = 45.0
t0 = time.time()
i = 0
with torch.no_grad():
    while time.time() - t0 < PREHEAT_S:
        runners[i % len(runners)].infer(x, prompt="")
        i += 1
print("[preheat] %.0f s, %d iters" % (time.time() - t0, i), flush=True)

ts = {n: [] for n in names}
with torch.no_grad():
    for rnd in range(3):
        for n, R in zip(names, runners):
            for _ in range(20):
                t = time.perf_counter(); R.infer(x, prompt=""); ts[n].append((time.perf_counter() - t) * 1e3)

print("\n%-52s %9s %9s %9s %9s" % ("model", "median", "min", "p90", "n"))
for n in names:
    s = sorted(ts[n])
    print("%-52s %9.2f %9.2f %9.2f %9d" % (n, statistics.median(s), s[0], s[int(0.9 * len(s))], len(s)))
