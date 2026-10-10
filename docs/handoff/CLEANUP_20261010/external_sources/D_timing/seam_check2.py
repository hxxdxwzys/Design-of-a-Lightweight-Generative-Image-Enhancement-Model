# -*- coding: utf-8 -*-
"""拼缝检测 v3 严格版 (2026-09-11): 用【邻域基线】排除图像内容偏置
对每个 512 整数倍位置 p: ratio = dx[p] / mean(dx[p-24:p-6] ∪ dx[p+6:p+24])
  真接缝 => ratio 显著 > 1 且仅出现在 p 处; 内容偏置 => ratio ≈ 1
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from PIL import Image

PKGS = {
    'M24': Path(r'D:/models/final_compliant/output_dir'),
    'm0s1': Path(r'D:/models/final_compliant_14b/output_dir'),
    'LQ': Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'),
}
TILE = 512
N = 12


def profile_ratios(arr1d):
    """对每个 tile 边界返回 ratio 列表"""
    out = []
    for p in range(TILE, len(arr1d) - 30, TILE):
        base = np.concatenate([arr1d[p - 24:p - 5], arr1d[p + 6:p + 25]])
        if len(base) == 0 or base.mean() <= 0:
            continue
        out.append(float(arr1d[p] / base.mean()))
    return out


print('=== 严格拼缝检测（邻域基线; 1.0 = 无接缝）===')
for tag, d in PKGS.items():
    if not d.exists():
        print(f'[skip] {tag}'); continue
    cs, rs = [], []
    for f in sorted(d.glob('*.jpg'))[:N]:
        g = np.asarray(Image.open(f).convert('L'), np.float32)
        dx = np.abs(np.diff(g, axis=1)).mean(axis=0)
        dy = np.abs(np.diff(g, axis=0)).mean(axis=1)
        cs += profile_ratios(dx)
        rs += profile_ratios(dy)
    cs, rs = np.array(cs), np.array(rs)
    print(f'\n[{tag}] n={N} 图 | 垂直缝样本 {len(cs)} | 水平缝样本 {len(rs)}')
    print(f'  垂直缝 ratio: 均值 {cs.mean():.3f} 中位 {np.median(cs):.3f} '
          f'p75 {np.percentile(cs,75):.3f} p90 {np.percentile(cs,90):.3f} '
          f'占比>1.15: {100*np.mean(cs>1.15):.0f}%')
    print(f'  水平缝 ratio: 均值 {rs.mean():.3f} 中位 {np.median(rs):.3f} '
          f'p75 {np.percentile(rs,75):.3f} p90 {np.percentile(rs,90):.3f} '
          f'占比>1.15: {100*np.mean(rs>1.15):.0f}%')
    verdict = ('⚠️ 存在真实接缝' if (np.median(cs) > 1.03 or np.median(rs) > 1.03)
               else '✅ 无显著接缝（中位数接近 1）')
    print(f'  → {verdict}')
