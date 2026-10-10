# -*- coding: utf-8 -*-
import sys, io, re
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(r'C:\生成式图像增强轻量模型设计挑战')
for p in sorted(ROOT.glob('*.md'), key=lambda x: x.stat().st_size):
    txt = io.open(p, encoding='utf-8', errors='replace').read()
    lines = [l.rstrip() for l in txt.splitlines()]
    heads = [l for l in lines if l.lstrip().startswith('#')][:8]
    first = next((l.strip() for l in lines if l.strip()), '')
    print('=' * 78)
    print('%s  (%.1f KB, %d 行)' % (p.name, p.stat().st_size / 1024, len(lines)))
    if not heads:
        print('  首行:', first[:110])
    for h in heads:
        print('  ', h[:110])
