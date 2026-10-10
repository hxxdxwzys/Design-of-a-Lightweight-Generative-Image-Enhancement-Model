# -*- coding: utf-8 -*-
"""检查提交图/LQ 的 JPEG 参数: quality(推断) + 色度子采样 (2026-09-11)"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from PIL import Image

SOURCES = {
    'M24': r'D:/models/final_compliant/output_dir',
    'm0s1': r'D:/models/final_compliant_14b/output_dir',
    's097': r'D:/models/wave2_m0s7_s097/output_dir',
    'LQ测试集': r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集',
    'GT验证集': r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集',
}


def describe(p: Path):
    im = Image.open(p)
    layers = getattr(im, 'layer', None)
    if layers:
        sub = '+'.join(str(l[1]) for l in layers)
    else:
        sub = '?'
    return f'q={im.info.get("quality", "?")} 采样={sub} size={im.size}'


for tag, d in SOURCES.items():
    dd = Path(d)
    files = sorted(dd.glob('*.jpg'))[:4] if 'GT' not in tag else sorted(dd.glob('*_gt.*'))[:2]
    print(f'== {tag}')
    for p in files:
        try:
            print(f'   {p.name:<14} {describe(p)}')
        except Exception as e:
            print(f'   {p.name:<14} 读取失败 {e}')
