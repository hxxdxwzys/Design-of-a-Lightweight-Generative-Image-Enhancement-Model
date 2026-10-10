r"""剖析 V4 包的 512 前向：找"零质量成本的时延空间"。
分别测 (a) CUDA Graph 重放路径（评测 harness 实际走的）(b) 直接调用 traced 模块（不含图）。
"""
import importlib.util, os, sys, time
from pathlib import Path
import torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MD = r"D:\models\pkg_TinyRefV4\model_dir"
spec = importlib.util.spec_from_file_location("r", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(MD)
dev = R.device
x = torch.rand(1, 3, 512, 512, device=dev) * 2 - 1
xf = x.half()

print("graph 可用:", R.graph is not None, "| static_in:", tuple(R.static_in.shape) if R.static_in is not None else None)

def bench(fn, n=50, warm=10):
    with torch.no_grad():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1e3)
    ts.sort()
    return ts[len(ts)//2], ts[0]

med_g, min_g = bench(lambda: R.infer(x, prompt=""))
print("(a) 完整 runner（含图）: median %.2f ms  min %.2f ms" % (med_g, min_g))

med_m, min_m = bench(lambda: R.m(xf))
print("(b) 仅 traced 模块（no graph, fp16 输入）: median %.2f ms  min %.2f ms" % (med_m, min_m))

from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    for _ in range(3): R.m(xf)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(5): R.m(xf)
        torch.cuda.synchronize()
print()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22))
