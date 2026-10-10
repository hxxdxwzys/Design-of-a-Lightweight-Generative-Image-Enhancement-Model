# -*- coding: utf-8 -*-
"""全仓清理残留占位符 %%BT%% -> 反引号 (只影响含占位符的 md)."""
import io, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BT = chr(96)
ROOT = Path(r'C:\生成式图像增强轻量模型设计挑战')
for p in sorted(ROOT.rglob('*.md')):
    if '.git' in p.parts:
        continue
    t = io.open(p, encoding='utf-8').read()
    if '%%BT%%' in t:
        n = t.count('%%BT%%')
        io.open(p, 'w', encoding='utf-8').write(t.replace('%%BT%%', BT))
        print('[fixed] %-46s %d 处' % (str(p.relative_to(ROOT)), n))
print('--- 复查 ---')
left = [str(p.relative_to(ROOT)) for p in ROOT.rglob('*.md') if '%%BT%%' in io.open(p, encoding='utf-8', errors='ignore').read()]
print('残留:', left if left else '无')
