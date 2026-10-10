# -*- coding: utf-8 -*-
"""最小化复测 1.4B(wave2) runner 的 512 时延: 只做 3 次前向, 逐次打印"""
import sys, time, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
import numpy as np
from PIL import Image

for md in sys.argv[1:]:
    md = Path(md)
    print('=== %s ===' % md, flush=True)
    if not (md / 'runner.py').exists():
        print('  !! 无 runner.py, 跳过', flush=True); continue
    sys.path.insert(0, str(md))
    try:
        for m in [k for k in list(sys.modules) if k == 'runner']:
            del sys.modules[m]
        import runner as R
        torch.cuda.synchronize(); t0 = time.time()
        run = R.Runner(str(md))
        torch.cuda.synchronize(); print('  载入 %.1f s' % (time.time() - t0), flush=True)
        x = torch.rand(1, 3, 512, 512, device='cuda') * 2 - 1
        for i in range(3):
            torch.cuda.synchronize(); t1 = time.time()
            with torch.no_grad():
                y = run.infer(x, prompt='')
            torch.cuda.synchronize(); dt = (time.time() - t1) * 1000
            o = y[0] if isinstance(y, (tuple, list)) else y
            print('  第%d次 %.1f ms  out=%s std=%.4f' % (i + 1, dt, tuple(o.shape), float(o.std())), flush=True)
        del run
        torch.cuda.empty_cache()
    except Exception as e:
        print('  EXC %s: %s' % (type(e).__name__, str(e)[:300]), flush=True)
    sys.path.pop(0)
print('LAT_W2_DONE', flush=True)
