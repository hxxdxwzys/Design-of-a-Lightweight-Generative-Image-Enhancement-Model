import re, io, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE = Path(r'D:/models/newsrc')
KEYS = re.compile(r'huggingface|hf_hub|download|pretrained|ckpt|weight|checkpoint|\.pth|\.safetensors|base model|SD2|SDXL|flux|qwen|upscale|x4|4x|scale factor|params', re.I)
out = []
for d in sorted(BASE.glob('*_code')):
    rp = d / 'README.md'
    if not rp.exists():
        rp = d / 'README_origin.md'
    out.append('=' * 70)
    out.append(f'### {d.name}')
    if not rp.exists():
        out.append('  (无 README)')
        continue
    lines = rp.read_text(encoding='utf-8', errors='replace').splitlines()
    hits = [i for i, l in enumerate(lines) if KEYS.search(l)]
    shown = set()
    for i in hits[:22]:
        for j in range(max(0, i - 1), min(len(lines), i + 2)):
            if j not in shown:
                shown.add(j)
                t = lines[j].strip()
                if t:
                    out.append(f'  {j+1}: {t[:170]}')
    w = d / 'weights'
    if w.exists():
        out.append('  [weights 目录] ' + str([p.name for p in w.iterdir()][:12]))
Path(r'D:/tmp_timing/newsrc_readmes.txt').write_text('\n'.join(out), encoding='utf-8')
print('\n'.join(out))
