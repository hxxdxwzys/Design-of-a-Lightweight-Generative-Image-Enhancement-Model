# -*- coding: utf-8 -*-
"""v2 被打断在 37200/40000 步 -> 用 s36000 直接走 productize"""
import sys
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
from night_final import productize, log

log('=== v2 手动接续 productize (从被超时打断的训练) ===')
r = productize(Path(r'D:/models/tiny_v1/v2_s36000.pt'), 'TinyV2', 2, '256 域 0.87M, 目标 ~24x')
log(f'结果: {r}')
log('PRODUCTIZE_V2_DONE')
