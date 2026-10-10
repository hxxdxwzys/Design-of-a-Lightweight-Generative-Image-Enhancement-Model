# -*- coding: utf-8 -*-
import sys, io
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(r'C:\生成式图像增强轻量模型设计挑战')
for name in ['research_diffusion_candidates_2026.md','STRATEGY_v9.md','NEXT_ROUND_PROMPT_20260911.md','SUBMIT_RUNBOOK_20260909.md','新世界模型分数对应表.md','HANDOFF.md','README.md']:
    p = ROOT / name
    txt = io.open(p, encoding='utf-8', errors='replace').read()
    lines = txt.splitlines()
    heads = [l for l in lines if l.lstrip().startswith('#')]
    print('=' * 76)
    print('%s  (%.1f KB, %d 行, %d 个标题)' % (name, p.stat().st_size/1024, len(lines), len(heads)))
    for h in heads[:26]:
        print('  ', h[:105])
    if len(heads) > 26: print('   ... 其余 %d 个标题省略' % (len(heads)-26))
