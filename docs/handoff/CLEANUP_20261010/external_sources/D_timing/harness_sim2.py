# -*- coding: utf-8 -*-
"""P0a: M24 合规 zip harness 模拟重验 (2026-09-08, future 修复后)
模拟官方: 解压 zip -> 在 model_dir 父目录外 import runner -> 512 infer
"""
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ZIP = Path(r'D:/models/Track2_CompliantM24.zip')
TMP = Path(r'D:/models/_harness_sim')


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir()
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(TMP)
    md = TMP / 'model_dir'
    print(f'[1] 解压 OK: output_dir={len(list((TMP/"output_dir").glob("*.jpg")))} model_dir={len(list(md.rglob("*")))}')

    # 模拟官方 harness: cwd 不在 model_dir 内, 纯净 sys.path
    sys.path.insert(0, str(md))
    sys.path = [p for p in sys.path if 'tmp_timing' not in p]
    t0 = time.time()
    import runner as R
    print(f'[2] runner import OK ({time.time()-t0:.1f}s)')
    r = R.Runner(str(md))
    x = torch.randn(1, 3, 512, 512)
    t1 = time.time()
    with torch.no_grad():
        y = r.infer(x)
    dt = time.time() - t1
    print(f'[3] infer OK: {tuple(y.shape)} dtype={y.dtype} range=({y.min().item():.2f},{y.max().item():.2f}) {dt:.2f}s')
    # 输出统计 (非 NaN/常数)
    assert torch.isfinite(y).all(), 'NaN in output'
    print(f'[4] 输出校验 PASS (finite, std={y.std().item():.4f})')
    print('HARNESS_SIM_PASS')


if __name__ == '__main__':
    main()
