# -*- coding: utf-8 -*-
"""runner 计时 v2 (2026-09-11): 正确同步 + 预热 + 分位数
用法: python time_runner2.py <model_dir> [--no-dynamo]
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

if '--no-dynamo' in sys.argv:
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    print('[cfg] TORCHDYNAMO_DISABLE=1')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
os.chdir(r'C:/models/finals')

dirs = [Path(a) for a in sys.argv[1:] if not a.startswith('--')] or \
       [Path(r'D:/models/final_compliant/model_dir')]


def bench(md: Path):
    import torch
    sys.path.insert(0, str(md))
    for m in list(sys.modules):
        if m == 'runner' or m.split('.')[0] in ('models', 'dinov2', 'vae_shim',
                                                'timm', 'einops', 'diffusers', 'fairscale'):
            del sys.modules[m]
    print(f'===== {md}', flush=True)
    t0 = time.time()
    import runner as R
    t_imp = time.time() - t0
    t0 = time.time()
    obj = R.Runner(str(md))
    t_ctor = time.time() - t0
    torch.cuda.synchronize()
    x = torch.rand(1, 3, 512, 512) * 2 - 1
    # 预热 3 次
    t_first = None
    for i in range(3):
        t0 = time.time()
        with torch.no_grad():
            obj.infer(x)
        torch.cuda.synchronize()
        if i == 0:
            t_first = time.time() - t0
    # 正式 10 次
    ts = []
    for i in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            obj.infer(x)
        torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000)
    ts_sorted = sorted(ts)
    print(f'  import runner      : {t_imp*1000:8.0f} ms', flush=True)
    print(f'  Runner 构造(含load): {t_ctor*1000:8.0f} ms', flush=True)
    print(f'  首次 infer(含预热) : {t_first*1000:8.0f} ms', flush=True)
    print(f'  同步后 10 次 infer : ' + ' '.join(f'{v:.0f}' for v in ts) + ' ms', flush=True)
    print(f'  中位 {ts_sorted[4]:.0f} ms | 最小 {ts_sorted[0]:.0f} | p90 {ts_sorted[8]:.0f} | '
          f'最大 {ts_sorted[-1]:.0f} | 抖动 {ts_sorted[-1]/max(ts_sorted[0],1):.2f}x', flush=True)
    print(f'  ★ 构造/中位 = {t_ctor*1000/ts_sorted[4]:.1f}x ; '
          f'冷启动(import+构造+首次)/中位 = {(t_imp+t_ctor+t_first)*1000/ts_sorted[4]:.1f}x',
          flush=True)
    sys.path.remove(str(md))


for d in dirs:
    if (d / 'runner.py').exists():
        bench(d)
