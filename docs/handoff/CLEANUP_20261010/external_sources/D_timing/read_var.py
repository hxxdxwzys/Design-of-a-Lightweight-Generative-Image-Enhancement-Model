import sys, subprocess, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# ① VARestorer README 关键段
rp = Path(r'D:/models/newsrc/VARestorer_code/README.md')
t = rp.read_text(encoding='utf-8', errors='replace')
print('=== VARestorer README (前 90 行) ===')
print('\n'.join(t.splitlines()[:90])[:2600])
print()
# ② 各仓库的 base model / 权重线索
import re
for name in ('VARestorer_code', 'TADSR_code', 'PixRestore_code'):
    d = Path(r'D:/models/newsrc') / name
    hits = []
    for f in list(d.rglob('*.sh')) + list(d.rglob('*.md')) + list(d.rglob('*.yaml')) + list(d.rglob('*.py')):
        try:
            s = f.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        for m in re.finditer(r'[\w\-/\.]*(huggingface\.co|hf_hub_download|snapshot_download)[^\s\)\x27"]*', s):
            hits.append(f'{f.relative_to(d)}: {m.group(0)[:120]}')
        if len(hits) > 12:
            break
    print(f'--- {name} 的 HF 引用 ---')
    print('\n'.join(hits[:10]) if hits else '  (无)')
print('VAR_DONE')
