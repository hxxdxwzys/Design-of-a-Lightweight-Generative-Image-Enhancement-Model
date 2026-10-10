# -*- coding: utf-8 -*-
import sys, numpy as np
from pathlib import Path
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
CASES = ['case1','case7','case12','case30','case54','case77']
M24 = Path(r'D:/models/final_compliant/output_dir')
def arr(p):
    return np.asarray(Image.open(p).convert('RGB')).astype(np.float32)
for tag in ('A_base','C_vae2048'):
    ds = []
    for c in CASES:
        p = Path(r'D:/models/probe_render')/tag/(c+'.png')
        if not p.exists(): continue
        a = arr(p); b = arr(M24/(c+'.jpg'))
        if a.shape != b.shape:
            print(tag, c, 'shape', a.shape, b.shape); continue
        ds.append((c, np.abs(a-b).mean(), np.abs(a-b).max()))
    if ds:
        print('%-10s 与已提交 M24 件相比: 平均差 %s  (均值 %.2f/255)' % (
            tag, ' '.join('%s=%.1f' % (c, d) for c, d, _ in ds), float(np.mean([d for _, d, _ in ds]))))
