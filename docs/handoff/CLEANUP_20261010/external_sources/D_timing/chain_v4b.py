# -*- coding: utf-8 -*-
"""v4b: 128 域 + 目标域快速退化 -> 训练 -> 自动 productize"""
import sys
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
from night_final import run, productize, log
PY = r'D:/Python/python.exe'
log('=== v4b 启动 (128 域, 目标域退化, bs=16) ===')
run('train v4b', [PY, '-u', r'D:/tmp_timing/train_tiny.py', '--res', '512', '--down', '4',
                  '--deg_small', '1', '--bs', '16', '--steps', '40000', '--tag', 'v4'], timeout=6 * 3600)
ck = Path(r'D:/models/tiny_v1/v4_s40000.pt')
if not ck.exists():
    cks = sorted(Path(r'D:/models/tiny_v1').glob('v4_s*.pt'), key=lambda p: int(p.stem.split('_s')[-1]))
    ck = cks[-1] if cks else None
if ck:
    productize(ck, 'TinyV4', 4, '128 域 0.87M, 目标 ~78x (激进提速)')
log('V4B_CHAIN_DONE')
