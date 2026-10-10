# -*- coding: utf-8 -*-
"""验证打包后的 TinyV1 runner: 解 zip -> 官方调用方式 -> 计时 (GPU 上有 v2 训练, 读数偏高)"""
import io, sys, time, zipfile
from pathlib import Path
import torch

Z = r'D:/models/upload/Track2_TinyV1.zip'
tmp = Path(r'D:/tmp_timing/_tinyv1_verify'); tmp.mkdir(exist_ok=True)
with zipfile.ZipFile(Z) as z:
    for n in z.namelist():
        if n.startswith('model_dir/'):
            z.extract(n, tmp)
md = tmp / 'model_dir'
sys.path.insert(0, str(md))
import runner as R
t0 = time.time(); r = R.Runner(str(md)); print(f'load {time.time()-t0:.1f}s', flush=True)
x = torch.randn(1, 3, 512, 512).clamp(-1, 1)
with torch.no_grad():
    for _ in range(5):
        y = r.infer(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(50):
        y = r.infer(x)
    torch.cuda.synchronize()
ms = (time.perf_counter() - t) / 50 * 1000
print(f'[packaged runner] {ms:.2f} ms -> {285.6/ms:.2f}x | out {tuple(y.shape)} std={y.std().item():.3f} nan={bool(torch.isnan(y).any())}')
print('VERIFY_DONE')
