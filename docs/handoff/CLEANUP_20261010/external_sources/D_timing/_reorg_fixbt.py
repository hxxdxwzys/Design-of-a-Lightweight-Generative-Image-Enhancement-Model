# -*- coding: utf-8 -*-
"""把新文档里的 %%BT%% 占位符还原为反引号, 并把裸的代码块语言行还原为三反引号围栏."""
import io, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(r'C:\生成式图像增强轻量模型设计挑战')
TARGETS = ['AGENTS.md', 'docs/project_map.md', 'docs/architecture.md', 'docs/current_task.md',
           'docs/decisions.md', 'docs/progress.md', 'docs/review.md']
BT = chr(96)
FENCE = BT * 3
LANGS = {'cpp', 'py', 'bash', 'json', 'python', 'powershell', 'ps1', 'text', 'yaml'}
for rel in TARGETS:
    p = ROOT / rel
    if not p.exists():
        print('[missing]', rel); continue
    txt = io.open(p, encoding='utf-8').read()
    n_bt = txt.count('%%BT%%')
    txt = txt.replace('%%BT%%', BT)
    out, n_fence = [], 0
    for line in txt.splitlines():
        if line.strip() in LANGS and not line.startswith(BT):
            out.append(FENCE); n_fence += 1
        else:
            out.append(line)
    io.open(p, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    left = '\n'.join(out).count('%%BT%%')
    print('%-26s 反引号 %4d 处, 围栏 %2d 处, 残留占位符 %d' % (rel, n_bt, n_fence, left))
# 全仓扫描残留
bad = []
for p in ROOT.rglob('*.md'):
    if '.git' in p.parts: continue
    try:
        if '%%BT%%' in io.open(p, encoding='utf-8', errors='ignore').read():
            bad.append(str(p.relative_to(ROOT)))
    except Exception: pass
print('全仓残留 %%BT%% 的文件:', bad if bad else '无')
