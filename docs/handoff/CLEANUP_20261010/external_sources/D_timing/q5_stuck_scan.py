# -*- coding: utf-8 -*-
"""扫描 M24 提交图 (与 LQ) 的极端像素/局部爆点, 找可能的黑洞/白斑."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
M24D = Path(r'D:/models/final_compliant/output_dir')
rows=[]
for i in range(1, 101):
    n = 'case%d' % i
    a = np.asarray(Image.open(M24D/(n+'.jpg')).convert('RGB')).astype(np.int16)
    b = np.asarray(Image.open(TEST/(n+'.jpg')).convert('RGB')).astype(np.int16)
    g = a.mean(2); gb = b.mean(2)
    p0 = float((g < 1.5).mean()*100); p255 = float((g > 253.5).mean()*100)
    q0 = float((gb < 1.5).mean()*100); q255 = float((gb > 253.5).mean()*100)
    # 8x8 块级极值: 找局部整块近黑/近白
    H,W = g.shape; bh, bw = H//8, W//8
    blocks = g[:bh*8,:bw*8].reshape(bh,8,bw,8).mean((1,3))
    dark = int((blocks < 12).sum()); bright = int((blocks > 244).sum())
    rows.append((n, a.shape, p0, p255, q0, q255, dark, bright, float(g.min()), float(g.max())))
rows.sort(key=lambda r: -(r[6]+r[7]))
print('%-8s %-13s %7s %7s | %7s %7s | %5s %5s' % ('case','shape','M24黑%','M24白%','LQ黑%','LQ白%','M24暗块','M24亮块'))
for r in rows[:12]:
    print('%-8s %-13s %7.3f %7.3f | %7.3f %7.3f | %5d %5d' % (r[0], str(r[1]), r[2], r[3], r[4], r[5], r[6], r[7]))
tot_dark = sum(r[6] for r in rows); tot_bright = sum(r[7] for r in rows)
print('总计: M24 暗块 %d / 亮块 %d (共 100 图 x 48 块 = 4800)' % (tot_dark, tot_bright))
print('M24 黑/白像素比例中位数: %.4f%% / %.4f%%  (LQ: %.4f%% / %.4f%%)' % (
    float(np.median([r[2] for r in rows])), float(np.median([r[3] for r in rows])),
    float(np.median([r[4] for r in rows])), float(np.median([r[5] for r in rows]))))
print('SCAN_DONE')
