# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TILE = 512
def seam(d):
    cs, rs = [], []
    for f in sorted(Path(d).glob('*.png'))[:3]:
        g = np.asarray(Image.open(f).convert('L'), np.float32)
        if g.shape[1] < TILE * 2:
            continue
        dx = np.abs(np.diff(g, axis=1)).mean(axis=0); dy = np.abs(np.diff(g, axis=0)).mean(axis=1)
        for arr, acc in ((dx, cs), (dy, rs)):
            for p in range(TILE, len(arr) - 30, TILE):
                b = np.concatenate([arr[p-24:p-5], arr[p+6:p+25]])
                if len(b) and b.mean() > 0:
                    acc.append(float(arr[p] / b.mean()))
    return (float(np.median(cs)) if cs else float('nan'), float(np.median(rs)) if rs else float('nan'),
            len(cs), (np.asarray(Image.open(sorted(Path(d).glob('*.png'))[0])).shape if cs else None))
for tag, d in [('seed_ensemble/s108','D:/models/seed_ensemble_test/s108'),
               ('seed_ensemble/s208','D:/models/seed_ensemble_test/s208'),
               ('seed_ensemble/s308','D:/models/seed_ensemble_test/s308'),
               ('seed_ensemble/avg4','D:/models/seed_ensemble_test/avg4'),
               ('sweep/pred_sc0.8','D:/models/sweep_scale_val/pred_sc0.8'),
               ('sweep/pred_sc0.0','D:/models/sweep_scale_val/pred_sc0.0')]:
    if not Path(d).exists():
        print('%-22s (缺)' % tag); continue
    a, b, n, shp = seam(d)
    print('%-22s 竖缝 %s  横缝 %s  (n=%d, shape=%s)' % (tag, ('%.3f' % a) if a == a else 'nan', ('%.3f' % b) if b == b else 'nan', n, shp))
print('SEAM_VAL_DONE')
