# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from PIL import Image
TEST = Path('C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
idx = [''.join(c for c in f.stem if c.isdigit()) for f in sorted(TEST.glob('*.jpg'))[:12]]
SETS = [('LQ', TEST), ('pkg_TinyV4g', Path('D:/models/pkg_TinyV4g/output_dir')),
        ('M24', Path('D:/models/final_compliant/output_dir'))]
print('%-14s %9s %8s %8s %8s' % ('set', 'lapvar', 'std', 'mean', '|out-LQ|'))
lqs = []
for i in idx:
    p = TEST / ('case%s.jpg' % i)
    if p.exists():
        a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float64)
        lqs.append(a)
for name, d in SETS:
    lv, sd, mn, dl = [], [], [], []
    for j, i in enumerate(idx):
        p = d / ('case%s.jpg' % i)
        if not p.exists() or j >= len(lqs):
            continue
        a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float64)
        g = a.mean(2)
        lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
        lv.append(lap.var()); sd.append(g.std()); mn.append(g.mean())
        dl.append(float(np.abs(a - lqs[j]).mean()))
    print('%-14s %9.1f %8.2f %8.2f %8.2f  (n=%d)' % (name, np.mean(lv), np.mean(sd), np.mean(mn), np.mean(dl), len(lv)))
print('SETSTATS2_DONE')
