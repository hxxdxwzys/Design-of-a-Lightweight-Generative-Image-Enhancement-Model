# -*- coding: utf-8 -*-
"""runner 计时拆解 (2026-09-11): 构造 / 首次推理 / 稳态推理
盲区: 若官方计时包含 Runner(model_dir) 构造或首次调用, 我们的 torch.load(GB 级) 与
      dynamo 编译尝试会贡献数秒 -> 加速比灾难
用法: python time_runner.py <model_dir> [more]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

os.chdir(r'C:/models/finals')


def bench(md: Path):
    import torch
    sys.path.insert(0, str(md))
    for m in list(sys.modules):
        if m == 'runner' or m.split('.')[0] in ('models', 'dinov2', 'vae_shim',
                                                'timm', 'einops', 'diffusers', 'fairscale'):
            del sys.modules[m]
    print(f'===== {md}')
    t0 = time.time()
    import runner as R
    t_import = time.time() - t0
    t0 = time.time()
    obj = R.Runner(str(md))
    t_ctor = time.time() - t0
    x = torch.rand(1, 3, 512, 512) * 2 - 1
    ts = []
    for i in range(6):
        t0 = time.time()
        with torch.no_grad():
            obj.infer(x)
        ts.append(time.time() - t0)
    print(f'  import runner : {t_import*1000:8.0f} ms')
    print(f'  Runner(构造)  : {t_ctor*1000:8.0f} ms   <-- 含 torch.load 权重')
    print(f'  首次 infer    : {ts[0]*1000:8.0f} ms   <-- 含 dynamo/首次分配')
    print(f'  稳态 infer    : {ts[1]*1000:8.0f} / {ts[2]*1000:.0f} / {ts[3]*1000:.0f} / '
          f'{ts[4]*1000:.0f} / {ts[5]*1000:.0f} ms  中位 {sorted(ts[1:])[2]*1000:.0f} ms')
    total_first = (t_import + t_ctor + ts[0])
    print(f'  ★ 冷启动总计(import+构造+首次) = {total_first*1000:.0f} ms  '
          f'= 稳态的 {total_first/sorted(ts[1:])[2]:.1f} 倍')
    sys.path.remove(str(md))


if __name__ == '__main__':
    dirs = [Path(a) for a in sys.argv[1:]] or [
        Path(r'D:/models/final_compliant/model_dir'),
        Path(r'D:/models/final_compliant_14b/model_dir'),
    ]
    for d in dirs:
        if (d / 'runner.py').exists():
            bench(d)
