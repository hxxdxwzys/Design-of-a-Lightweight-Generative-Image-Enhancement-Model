# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
h = Path(r'C:/生成式图像增强轻量模型设计挑战/HANDOFF.md')
s = Path(r'D:/tmp_timing/_sec22.md').read_text(encoding='utf-8')
t = h.read_text(encoding='utf-8')
if '§22' in t:
    print('§22 already present, skip')
else:
    h.write_text(t.rstrip('\n') + '\n' + s, encoding='utf-8')
    print('appended §22; new len', len(t) + len(s))
