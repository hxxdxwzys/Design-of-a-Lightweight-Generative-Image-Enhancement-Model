# -*- coding: utf-8 -*-
"""Q2 热点定位 (2026-09-15): M24 runner 352.9 ms 花在哪?
用 torch.profiler 取 CUDA op 级表; 再按四段粗分 (VAE enc / dinov2 / DiT / decoder) 计时。
只读不改 —— 提速方案等这张表出来再定。
"""
import sys, time, torch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
MD = 'D:/models/final_compliant/model_dir'
sys.path.insert(0, MD)
import runner as R

run = R.Runner(MD)
x = ((torch.rand(1, 3, 512, 512) * 2 - 1)).cuda()
with torch.no_grad():
    for _ in range(6):
        y = run.infer(x)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(6):
            y = run.infer(x)
        torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=22), flush=True)
print()
print('=== 整链计时 (逐次 sync, median of 20) ===')
import statistics
ts = []
with torch.no_grad():
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run.infer(x)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
print('median %.1f ms | min %.1f | mean %.1f' % (statistics.median(ts), min(ts), sum(ts) / len(ts)))
print('Q2_PROFILE_DONE')
