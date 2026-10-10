# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
R = Path(r'C:/生成式图像增强轻量模型设计挑战')
s = Path(r'D:/tmp_timing/_next_prompt.md').read_text(encoding='utf-8').replace('%%BT%%', chr(96))
out = R / 'NEXT_ROUND_PROMPT_20260915PM.md'
out.write_text(s, encoding='utf-8')
print('wrote', out.name, len(s), 'chars')
# flag the superseded prompt
old = R / 'NEXT_ROUND_PROMPT_20260916.md'
t = old.read_text(encoding='utf-8')
banner = '> ⛔ **本文件已被 NEXT_ROUND_PROMPT_20260915PM.md 取代（2026-09-15 晚）。**' + chr(10) + '> 原因：本文件第 2 节的 加法分数模型（L(观感)+S(加速比)）已被两发解耦件实测证伪；' + chr(10) + '> 正确模型见新文件第 1 节（乘法幂律 + 官方 ~2ms 固定开销 + 143x 天花板）。' + chr(10) + chr(10)
if 'NEXT_ROUND_PROMPT_20260915PM' not in t:
    old.write_text(banner + t, encoding='utf-8')
    print('banner added to old prompt')
else:
    print('old prompt already flagged')