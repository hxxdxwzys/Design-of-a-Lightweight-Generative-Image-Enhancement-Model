# -*- coding: utf-8 -*-
"""把误插到 20.5 之前的 §21 挪到文件末尾"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
p = Path(r'C:/生成式图像增强轻量模型设计挑战/HANDOFF.md')
t = p.read_text(encoding='utf-8')
i = t.index('## §21 2026-09-15 中午~下午')
j = t.index('### 20.5 挂起项（用户手上）')
assert i < j, (i, j)
blk = t[i:j]
t2 = t[:i] + t[j:]
if not t2.endswith('\n'):
    t2 += '\n'
t2 += '\n' + blk
p.write_text(t2, encoding='utf-8')
print('moved §21 to end; len %d -> %d' % (len(t), len(t2)))
print('tail:', t2[-120:].replace('\n', ' | '))
