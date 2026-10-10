# -*- coding: utf-8 -*-
"""v4: 128 域 (down=4, 再快 4x, 目标 ~90x) —— 等 v2 的 productize 结束后自动开训"""
import sys, time
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
from night_final import run, productize, log, wait_marker, latest_ckpt
PY = r'D:/Python/python.exe'
log('=== v4 链启动: 等 TinyV2 productize 完成 ===')
wait_marker(Path(r'D:/tmp_timing/night_final.log'), 'PRODUCTIZE_V2_DONE', 2 * 3600)
log('v2 完成, 开始训练 v4 (128 域)')
run('train v4 (down=4)', [PY, '-u', r'D:/tmp_timing/train_tiny.py', '--res', '512', '--down', '4',
                          '--bs', '16', '--steps', '40000', '--tag', 'v4'], timeout=5 * 3600)
ck = latest_ckpt('v4')
if ck:
    productize(ck, 'TinyV4', 4, '128 域 0.87M, 目标 ~90x (激进提速)')
log('V4_CHAIN_DONE')
