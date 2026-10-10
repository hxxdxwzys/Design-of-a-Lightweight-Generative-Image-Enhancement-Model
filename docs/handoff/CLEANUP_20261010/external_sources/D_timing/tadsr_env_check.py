# -*- coding: utf-8 -*-
"""TADSR 依赖体检(只 import, 不跑模型)"""
import sys, traceback
REPO = r'D:/models/newsrc/TADSR_code'
sys.path.insert(0, REPO)
for m in ['tadsr', 'my_utils.wavelet_color_fix', 'ram.models.ram_lora', 'ram']:
    try:
        __import__(m)
        print('OK      ', m)
    except Exception as e:
        print('FAIL    ', m, '->', type(e).__name__, e)
        tb = traceback.format_exc().strip().splitlines()[-3:]
        for l in tb:
            print('        ', l.strip()[:150])
