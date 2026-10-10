# -*- coding: utf-8 -*-
"""q3_runner_bench.py — 拆解 runner 每次 infer 的时延构成 (H2D / 重采样 / 图重放 / 输出算子)."""
import os, sys, time, importlib.util
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
torch.backends.cudnn.benchmark = True

def load_runner(pkg):
    p = Path(pkg) / 'model_dir' / 'runner.py'
    spec = importlib.util.spec_from_file_location('rf_' + Path(pkg).name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Runner(str(Path(pkg) / 'model_dir'))

def bench(fn, n=200, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return float(np.median(a)), float(a.mean()), float(a.min())

x_cpu = torch.randn(1, 3, 512, 512)
x_gpu = x_cpu.cuda().half()

for pkg in (r'D:/models/pkg_TinyV4g', r'D:/models/pkg_TinyV7gan', r'D:/models/pkg_TinyV9b', r'D:/models/pkg_TinyV11b'):
    try:
        R = load_runner(pkg)
        nat = getattr(R, 'static_in', None)
        print('===== %s  NATIVE=%s graph=%s' % (Path(pkg).name, tuple(nat.shape[-2:]) if nat is not None else None, R.graph is not None))
        print('   infer(cpu)     med=%.3f mean=%.3f min=%.3f ms' % bench(lambda: R.infer(x_cpu)))
        print('   infer(cuda fp16) med=%.3f mean=%.3f min=%.3f ms' % bench(lambda: R.infer(x_gpu)))
        # 图重放本身
        if R.graph is not None:
            print('   graph.replay() med=%.3f mean=%.3f min=%.3f ms' % bench(lambda: R.graph.replay()))
            print('   static_in.copy_ med=%.3f ms' % bench(lambda: R.static_in.copy_(torch.randn_like(R.static_in).half()))[0])
        # H2D 本身
        print('   x_cpu->cuda fp16 med=%.3f ms' % bench(lambda: x_cpu.to('cuda', dtype=torch.float16))[0])
        # 重采样
        n = nat.shape[-1] if nat is not None else 256
        y = torch.randn(1, 3, n, n, device='cuda', dtype=torch.float16)
        print('   512->%d bicubic med=%.3f ms | %d->512 med=%.3f ms' % (
            n, bench(lambda: F.interpolate(x_gpu, size=(n, n), mode='bicubic', align_corners=False))[0],
            n, bench(lambda: F.interpolate(y, size=(512, 512), mode='bicubic', align_corners=False))[0]))
        yo = torch.randn(1, 3, 512, 512, device='cuda', dtype=torch.float16)
        print('   .float().clamp med=%.3f ms' % bench(lambda: yo.float().clamp(-1.0, 1.0))[0])
        del R
        torch.cuda.empty_cache()
    except Exception as e:
        print('=====', pkg, 'ERR', repr(e)[:300])
print('Q3_BENCH_DONE')
