# -*- coding: utf-8 -*-
"""文档整理第一步: 建目录 + 机械搬迁(不改内容). 目标文件不存在时跳过; 已存在则报告冲突."""
import io, os, subprocess, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(r'C:\生成式图像增强轻量模型设计挑战')
MOVES = {
    'docs/handoff': ['HANDOFF.md', 'NEXT_ROUND_PROMPT.md', 'NEXT_ROUND_PROMPT_20260904.md',
                     'NEXT_ROUND_PROMPT_20260909.md', 'NEXT_ROUND_PROMPT_20260910.md',
                     'NEXT_ROUND_PROMPT_20260911.md', 'NEXT_ROUND_PROMPT_20260913.md',
                     'NEXT_ROUND_PROMPT_20260915.md', 'NEXT_ROUND_PROMPT_20260915PM.md',
                     'NEXT_ROUND_PROMPT_20260915NIGHT.md', 'NEXT_ROUND_PROMPT_20260916.md',
                     'MORNING_NOTE_20260914.md', 'P0_AUDIT_20260914.md',
                     'Q_CAMPAIGN_20260915.md', 'CLAUDE.md'],
    'docs/reference': ['规则细则与判例台账.md', '新世界模型分数对应表.md',
                       'SUBMIT_RUNBOOK_20260909.md', 'RULE_UPDATE_20260902.md',
                       'FINAL_PLAN.md', 'research_diffusion_candidates_2026.md'],
    'docs/archive': ['STRATEGY_v9.md', 'BRAINSTORM_20260908.md', 'THEORETICAL_CEILING_FRAMEWORK.md',
                     'COMMERCIAL_API_PLAN.md', 'DOWNLOAD_NEWSRC.md',
                     'SERVER_SPEC.md', 'SERVER_PHASE0.md', 'SERVER_P100.md'],
}
for d in MOVES:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
# AGENTS.md 原始快照(改前留档, 保证零信息丢失)
snap = ROOT / 'docs' / 'handoff' / 'AGENTS_20260901_原始快照.md'
if not snap.exists():
    snap.write_bytes((ROOT / 'AGENTS.md').read_bytes())
    print('[snapshot] AGENTS.md ->', snap.relative_to(ROOT))
for d, names in MOVES.items():
    for n in names:
        src = ROOT / n
        if not src.exists():
            print('[skip-missing]', n); continue
        dst = ROOT / d / n
        if dst.exists():
            print('[CONFLICT exists]', dst.relative_to(ROOT)); continue
        r = subprocess.run(['git', 'mv', n, str(Path(d) / n)], cwd=str(ROOT),
                           capture_output=True, text=True, errors='replace')
        if r.returncode != 0:
            src.replace(dst)
            print('[moved-nontracked] %-42s -> %s' % (n, d))
        else:
            print('[git mv]           %-42s -> %s' % (n, d))
print('--- 剩余根目录 .md ---')
for p in sorted(ROOT.glob('*.md')):
    print('  ', p.name)
