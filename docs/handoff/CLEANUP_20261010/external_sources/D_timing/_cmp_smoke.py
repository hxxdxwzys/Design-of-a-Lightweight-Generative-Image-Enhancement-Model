# -*- coding: utf-8 -*-
import sys, numpy as np
from pathlib import Path
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
a = Path(r'D:/models/probe_render/smoke_base')
print('files:', [p.name for p in a.glob('*')][:5] if a.exists() else 'MISSING')
m24 = Path(r'D:/models/final_compliant/output_dir/case12.jpg')
for p in list(a.glob('*'))[:3]:
    try:
        A = np.asarray(Image.open(p).convert('RGB')).astype(np.float32)
        B = np.asarray(Image.open(m24).convert('RGB')).astype(np.float32)
        print(p.name, A.shape, 'B', B.shape)
        if A.shape == B.shape:
            print('  与 M24 件 case12 的平均差 %.3f/255  maxdiff %.1f' % (np.abs(A-B).mean(), np.abs(A-B).max()))
            g = A.mean(2); lap = (-4*g[1:-1,1:-1] + g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:])
            gb = B.mean(2); lapb = (-4*gb[1:-1,1:-1] + gb[:-2,1:-1] + gb[2:,1:-1] + gb[1:-1,:-2] + gb[1:-1,2:])
            print('  lapvar 新 %.1f / M24 %.1f' % (lap.var(), lapb.var()))
    except Exception as e:
        print(p, 'ERR', repr(e)[:150])
