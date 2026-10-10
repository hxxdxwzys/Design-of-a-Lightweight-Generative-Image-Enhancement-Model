# -*- coding: utf-8 -*-
"""尖峰定位 v3 (2026-09-11): 分别关掉 GC / dynamo / 预分配噪声, 看 2.2s 尖峰是否消失
用法: python time_runner3.py <model_dir> [--gc-off] [--no-dynamo] [--pre-z]
"""
from __future__ import annotations

import gc
import os
import statistics
import sys
import time
from pathlib import Path

GC_OFF = '--gc-off' in sys.argv
if '--no-dynamo' in sys.argv:
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
PRE_Z = '--pre-z' in sys.argv
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
os.chdir(r'C:/models/finals')

md = Path([a for a in sys.argv[1:] if not a.startswith('--')][0])
import torch  # noqa: E402
sys.path.insert(0, str(md))
import runner as R  # noqa: E402

obj = R.Runner(str(md))
x = torch.rand(1, 3, 512, 512) * 2 - 1
if GC_OFF:
    gc.disable()

# 预热
for _ in range(3):
    with torch.no_grad():
        obj.infer(x)
    torch.cuda.synchronize()

ts = []
for i in range(12):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        obj.infer(x)
    torch.cuda.synchronize()
    ts.append((time.time() - t0) * 1000)
if GC_OFF:
    gc.enable()
med = statistics.median(ts)
print(f'配置: gc_off={GC_OFF} dynamo_off={"--no-dynamo" in sys.argv}')
print('  12 次: ' + ' '.join(f'{v:.0f}' for v in ts))
print(f'  中位 {med:.0f} ms | 均值 {statistics.mean(ts):.0f} | 最大 {max(ts):.0f} | '
      f'尖峰(>1000ms)次数 {sum(1 for v in ts if v > 1000)}/12')
print(f'  ★ 若官方取【均值】: {statistics.mean(ts):.0f} ms ; 取【中位】: {med:.0f} ms ; '
      f'均值/中位 = {statistics.mean(ts)/med:.2f}x')
