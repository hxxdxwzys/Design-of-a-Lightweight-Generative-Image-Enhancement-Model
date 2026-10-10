# -*- coding: utf-8 -*-
"""拼缝检测 v2 (2026-09-11): 逐图计算, 避免不同尺寸混算
对每张图: 归一化梯度剖面, 取 512 整数倍处的值 -> 与 1.0 比较; 汇总跨图均值
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
}
TILE = 512
N = 12


def main():
    print('=== 拼缝检测（每图归一化：1.0 = 该图全局平均梯度）===')
    for tag, d in PKGS.items():
        if not d.exists():
            print(f'[skip] {tag}'); continue
        cs, rs = [], []
        for f in sorted(d.glob('*.jpg'))[:N]:
            g = np.asarray(Image.open(f).convert('L'), np.float32)
            dx = np.abs(np.diff(g, axis=1)).mean(axis=0)
            dy = np.abs(np.diff(g, axis=0)).mean(axis=1)
            cx = dx / dx.mean()
            ry = dy / dy.mean()
            cs += [float(cx[i]) for i in range(TILE, len(cx) - 8, TILE)]
            rs += [float(ry[i]) for i in range(TILE, len(ry) - 8, TILE)]
        cs, rs = np.array(cs), np.array(rs)
        print(f'\n[{tag}] 样本 {N} 张, 边界列 {len(cs)} 个, 边界行 {len(rs)} 个')
        print(f'  列(垂直缝) 边界梯度/全局: 均值 {cs.mean():.3f}  中位 {np.median(cs):.3f}  '
              f'p90 {np.percentile(cs, 90):.3f}  max {cs.max():.3f}')
        print(f'  行(水平缝) 边界梯度/全局: 均值 {rs.mean():.3f}  中位 {np.median(rs):.3f}  '
              f'p90 {np.percentile(rs, 90):.3f}  max {rs.max():.3f}')
        flag = '⚠️ 疑似接缝(边界梯度系统性偏高)' if (cs.mean() > 1.02 or rs.mean() > 1.02) else '✅ 无明显周期性接缝'
        print(f'  → {flag}')


if __name__ == '__main__':
    main()
