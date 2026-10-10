# -*- coding: utf-8 -*-
"""把 out_rest/result 里已跑完的 HYPIR 图校验并搬进 out_test100/result, 然后做全量对比"""
import shutil, subprocess, time, os, sys
from pathlib import Path
from PIL import Image

PY = r'D:/Python/python.exe'
RES = Path(r'D:/models/hypir/out_test100/result')
SRC = Path(r'D:/models/hypir/out_rest/result')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

RES.mkdir(parents=True, exist_ok=True)
moved = bad = 0
for p in sorted(SRC.glob('*.png')):
    try:
        with Image.open(p) as im:
            im.load()
    except Exception as e:
        print(f'  [残缺丢弃] {p.name}: {type(e).__name__}', flush=True)
        bad += 1
        continue
    shutil.move(str(p), str(RES / p.name))
    moved += 1
print(f'[moved] {moved} 张 (残缺 {bad}) -> 现在共 {len(list(RES.glob("*.png")))}/100', flush=True)

env = dict(os.environ, PYTHONIOENCODING='utf-8')
p = subprocess.run([PY, '-u', r'D:/tmp_timing/hypir_vs_m24.py'], capture_output=True,
                   text=True, encoding='utf-8', errors='replace', env=env, timeout=3600)
print((p.stdout or '') + (p.stderr or ''), flush=True)
print(f'[compare] rc={p.returncode}', flush=True)
print('HYPIR_MOVE_COMPARE_DONE', flush=True)
