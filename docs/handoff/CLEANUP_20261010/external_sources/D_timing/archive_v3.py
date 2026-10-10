# -*- coding: utf-8 -*-
"""归档 9-11 v3 五发: 重命名加 ——分数 并移入 提交历史/ (2026-09-11, 用户指令)"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

UP = Path(r'D:/models/upload')
HIST = Path(r'C:/生成式图像增强轻量模型设计挑战/提交历史')
# zip 名 -> 官方分 (v3 批, 用户提供)
SCORES = {
    'Track2_M24_v3.zip': 43.6951,
    'Track2_M0S1_SC093_v3.zip': 34.1056,
    'Track2_M0S7_SC10_v3.zip': 30.6253,
    'Track2_M0S7_SC097_v3.zip': 30.9363,
    'Track2_G3W18S1_SC093_v3.zip': 31.8185,
}

HIST.mkdir(parents=True, exist_ok=True)
print('=== 归档 (重命名 -> 提交历史/) ===')
for name, score in SCORES.items():
    src = UP / name
    if not src.exists():
        print(f'  !! 缺 {name}')
        continue
    stem = name[:-4]
    dst = HIST / f'{stem}——{score}.zip'
    if dst.exists():
        dst.unlink()
    sz = src.stat().st_size / 2**30
    shutil.move(str(src), str(dst))
    print(f'  移动 {name}  ->  {dst.name}   ({sz:.2f} GiB)')

print('\n=== upload/ 现状 ===')
rest = list(UP.glob('*'))
print('  (空)' if not rest else '\n'.join(f'  {p.name}' for p in rest))

print('\n=== 提交历史/ 现状 ===')
for p in sorted(HIST.iterdir()):
    if p.is_dir():
        n = len(list(p.rglob('*')))
        print(f'  [dir] {p.name}  ({n} 项)')
    else:
        print(f'  {p.stat().st_size/2**30:5.2f} GiB  {p.name}')
