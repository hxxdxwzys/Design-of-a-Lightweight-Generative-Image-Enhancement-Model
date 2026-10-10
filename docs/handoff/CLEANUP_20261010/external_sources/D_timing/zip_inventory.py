# -*- coding: utf-8 -*-
"""zip 全量清单 + 硬链接归并 (2026-09-11)
扫描所有相关目录, 按 (设备, inode) 归并 -> 一眼看清"同一份字节起了几个名字"
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOTS = [
    Path(r'D:/models'),
    Path(r'C:/生成式图像增强轻量模型设计挑战/提交历史'),
    Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace'),
    Path(r'C:/生成式图像增强轻量模型设计挑战'),
    Path(r'D:/tmp_timing'),
]
SKIP = ('_harness_zip_sim', '_audit_ab', 'pack_tmp', 'harness_sim')

rows = []
for root in ROOTS:
    if not root.exists():
        continue
    for p in root.rglob('*.zip'):
        s = str(p)
        if any(k in s for k in SKIP):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rows.append({'path': p, 'size': st.st_size, 'mtime': st.st_mtime,
                     'ino': (st.st_dev, st.st_ino), 'nlink': st.st_nlink})

groups = defaultdict(list)
for r in rows:
    groups[r['ino']].append(r)

print('=' * 100)
print('全部 zip（已按"同一份字节"归并；== 表示硬链接，删一个不影响另一个）')
print('=' * 100)
total_unique = 0
for ino, items in sorted(groups.items(), key=lambda kv: -max(x['size'] for x in kv[1])):
    sz = max(x['size'] for x in items)
    total_unique += sz
    tag = f'{sz/2**30:.2f} GiB'
    print(f'\n【{tag}】 同一个文件有 {len(items)} 个名字:')
    for it in sorted(items, key=lambda x: str(x['path'])):
        print(f'    {it["path"]}')
print()
print(f'去重后总占用: {total_unique/2**30:.2f} GiB')
print(f'条目总数    : {len(rows)}')
