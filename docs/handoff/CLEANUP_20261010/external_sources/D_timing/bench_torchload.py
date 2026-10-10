# -*- coding: utf-8 -*-
"""torch.load 加载方式对比 (2026-09-11): 普通 vs mmap vs 半精度预转换
目的: 把 Runner() 构造从 12s 压下来 (若官方计时含构造就是白捡的分)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch

P = Path(r'D:/models/final_compliant/model_dir/your_model.pt')
print(f'文件: {P}  {P.stat().st_size/2**30:.2f} GiB')
with open(P, 'rb') as f:
    magic = f.read(4)
print('文件头:', magic, '=>', 'zip 格式(可用 mmap)' if magic[:2] == b'PK' else 'legacy pickle(mmap 不可用)')

for name, kw in (('普通', {}), ('mmap=True', {'mmap': True})):
    try:
        t0 = time.time()
        pkg = torch.load(P, map_location='cpu', weights_only=False, **kw)
        dt = time.time() - t0
        n = len(pkg) if hasattr(pkg, '__len__') else '?'
        tot = sum(v.numel() * v.element_size() for v in pkg.values()
                  if isinstance(v, torch.Tensor)) / 2**30 if isinstance(pkg, dict) else 0
        print(f'  {name:<10}: {dt*1000:7.0f} ms | keys={n} | tensor 总量 {tot:.2f} GiB')
        # 首次访问一个张量, 看 mmap 是否惰性
        if isinstance(pkg, dict):
            k0 = next(iter(pkg))
            t0 = time.time()
            _ = pkg[k0].float().sum().item() if isinstance(pkg[k0], torch.Tensor) else None
            print(f'             首次访问第一个张量: {(time.time()-t0)*1000:.0f} ms')
        del pkg
    except Exception as e:
        print(f'  {name:<10}: 失败 {type(e).__name__}: {str(e)[:160]}')

print('\ntorch 版本:', torch.__version__)
