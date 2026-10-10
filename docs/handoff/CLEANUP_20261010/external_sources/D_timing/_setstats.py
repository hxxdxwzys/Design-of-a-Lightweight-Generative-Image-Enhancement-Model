# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from PIL import Image
TEST = Path('C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
idx = [''.join(c for c in f.stem if c.isdigit()) for f in sorted(TEST.glob('*.jpg'))[:12]]
SETS = [
    ('LQ', TEST),
    ('M24', Path('D:/models/final_compliant/output_dir')),
    ('w2_m0s1_s090', Path('D:/models/wave2_m0s1_s090/output_dir')),
    ('w2_m0s7_s097', Path('D:/models/wave2_m0s7_s097/output_dir')),
    ('w2_m0s1_s100', Path('D:/models/wave2_m0s1_s100/output_dir')),
    ('TinyV7gan', Path('D:/models/pkg_TinyV7gan/output_dir')),
    ('TinyV9b', Path('D:/models/pkg_TinyV9b/output_dir')),
    ('TinyV11b', Path('D:/models/pkg_TinyV11b/output_dir')),
]
print('%-14s %8s %8s %8s %8s %10s' % ('set', 'lapvar', 'std', 'mean', 'sat', 'HFratio'))
for name, d in SETS:
    lv, sd, mn, sa, hf = [], [], [], [], []
    for i in idx:
        p = d / ('case%s.jpg' % i)
        if not p.exists():
            continue
        a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float64)
        g = a.mean(2)
        lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
        lv.append(lap.var())
        sd.append(g.std()); mn.append(g.mean())
        mx = a.max(2); mnn = a.min(2)
        sa.append(float((mx - mnn).mean()))
        # 高频能量占比: 拉普拉斯能量 / 总能量
        hf.append(float((lap ** 2).mean() / (g.var() + 1e-9)))
    if lv:
        print('%-14s %8.1f %8.2f %8.2f %8.2f %10.4f  (n=%d)' % (
            name, np.mean(lv), np.mean(sd), np.mean(mn), np.mean(sa), np.mean(hf), len(lv)))
print('SETSTATS_DONE')
