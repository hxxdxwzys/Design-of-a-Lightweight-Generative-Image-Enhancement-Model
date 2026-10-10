# -*- coding: utf-8 -*-
"""zip 级 harness 模拟 (参数化): 解压 -> 纯净 import -> 512 infer
用法: python harness_sim_zip.py <zip路径>
"""
import shutil
import sys
import time
import zipfile
from pathlib import Path

import torch


def main():
    zip_path = Path(sys.argv[1])
    tmp = Path(r'D:/models/_harness_zip_sim')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    md = tmp / 'model_dir'
    outs = [n for n in zipfile.ZipFile(zip_path).namelist() if n.startswith('output_dir/')]
    print(f'[1] 解压 OK: output_dir={len(outs)} model_dir files={len(list(md.rglob("*")))}')
    sys.path.insert(0, str(md))
    t0 = time.time()
    import runner as R
    print(f'[2] import OK ({time.time()-t0:.1f}s)')
    r = R.Runner(str(md))
    x = torch.randn(1, 3, 512, 512)
    t1 = time.time()
    with torch.no_grad():
        y = r.infer(x)
    dt = time.time() - t1
    print(f'[3] infer OK: {tuple(y.shape)} range=({y.min().item():.2f},{y.max().item():.2f}) std={y.std().item():.3f} {dt:.1f}s')
    assert torch.isfinite(y).all() and y.std().item() > 0.05
    print('HARNESS_ZIP_SIM_PASS')


if __name__ == '__main__':
    main()
