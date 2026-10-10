# -*- coding: utf-8 -*-
"""探索收尾分析 (2026-09-09): 
1. 逐图 max 混合上界 (无参指标可选 -> 混合潜力)
2. m0s1 胜负图特征对比 (赢的图 vs 输的图)
3. 差值分布
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

RES = Path(r'D:/tmp_timing/nr_full100_clipiqa.json')
LQ = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
res = json.loads(RES.read_text(encoding='utf-8'))
m24, m0s1 = res['M24'], res['m0s1']
keys = sorted(m24)
d = {k: m24[k] - m0s1[k] for k in keys}

# 1) 混合上界
mix = {k: max(m24[k], m0s1[k]) for k in keys}
v24 = np.array([m24[k] for k in keys])
v01 = np.array([m0s1[k] for k in keys])
vmix = np.array([mix[k] for k in keys])
print(f'M24 mean={v24.mean():.4f} | m0s1 mean={v01.mean():.4f} | 逐图max混合 mean={vmix.mean():.4f}')
print(f'混合相对 M24 提升: {vmix.mean()-v24.mean():+.4f}')

# 2) 差值分布
dd = np.array(list(d.values()))
print(f'\n差值分布: mean={dd.mean():+.4f} p10={np.percentile(dd,10):+.4f} p50={np.percentile(dd,50):+.4f} p90={np.percentile(dd,90):+.4f}')
print(f'M24 胜(d>0): {(dd>0).sum()} | m0s1 胜: {(dd<0).sum()} | 平: {(dd==0).sum()}')

# 3) 胜负图特征 (LQ 图亮度/尺寸/对比度)
win01 = [k for k in keys if d[k] < 0]  # m0s1 赢
win24 = [k for k in keys if d[k] > 0.05]  # M24 大胜

def feat(k):
    p = LQ / f'{k}.jpg'
    if not p.exists():
        return None
    img = Image.open(p).convert('L')
    a = np.asarray(img, dtype=np.float32)
    return a.mean(), a.std(), img.size

print(f'\n=== m0s1 赢的 {len(win01)} 张 (LQ 亮度/对比度) ===')
for k in sorted(win01, key=lambda x: d[x])[:8]:
    f = feat(k)
    if f:
        print(f'  {k}: M24={m24[k]:.4f} m0s1={m0s1[k]:.4f} | LQ mean={f[0]:.1f} std={f[1]:.1f} size={f[2]}')

print(f'\n=== M24 大胜(>0.05)的图 ===')
for k in sorted(win24, key=lambda x: -d[x])[:8]:
    f = feat(k)
    if f:
        print(f'  {k}: M24={m24[k]:.4f} m0s1={m0s1[k]:.4f} | LQ mean={f[0]:.1f} std={f[1]:.1f} size={f[2]}')

# 4) 汇总特征对比
f01 = [feat(k) for k in win01 if feat(k)]
f24 = [feat(k) for k in win24 if feat(k)]
if f01 and f24:
    print(f'\nm0s1 赢图 LQ: mean亮度={np.mean([x[0] for x in f01]):.1f} 对比度={np.mean([x[1] for x in f01]):.1f}')
    print(f'M24 大胜图 LQ: mean亮度={np.mean([x[0] for x in f24]):.1f} 对比度={np.mean([x[1] for x in f24]):.1f}')
