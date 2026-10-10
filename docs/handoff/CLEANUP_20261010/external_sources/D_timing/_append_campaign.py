# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
NL = chr(10)
p = Path(r'C:/生成式图像增强轻量模型设计挑战/Q_CAMPAIGN_20260915.md')
t = p.read_text(encoding='utf-8')
s = Path(r'D:/tmp_timing/_q_campaign_add.md').read_text(encoding='utf-8')
if '## 6. 首轮筛选结果' not in t:
    p.write_text(t.rstrip() + NL + s, encoding='utf-8')
    print('appended section 6')
else:
    print('section 6 exists')