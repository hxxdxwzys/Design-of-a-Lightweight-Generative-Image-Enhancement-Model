# -*- coding: utf-8 -*-
"""下载候选新图像源的代码仓库 (2026-09-14)"""
import subprocess, sys, time
from pathlib import Path

DST = Path(r'D:/models/newsrc')
DST.mkdir(parents=True, exist_ok=True)
REPOS = [
    ('VARestorer_code', 'https://github.com/EternalEvan/VARestorer'),
    ('TADSR_code',      'https://github.com/zty557/TADSR'),
    ('ODTSR_code',      'https://github.com/RedMediaTech/ODTSR'),
    ('PixRestore_code', 'https://github.com/csslc/PixRestore'),
    ('TSD_SR_code',     'https://github.com/Microtreei/TSD-SR'),
]
env = dict(__import__('os').environ)
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    env.pop(k, None)

for name, url in REPOS:
    d = DST / name
    if d.exists() and any(d.iterdir()):
        print(f'[skip] {name} 已存在', flush=True)
        continue
    t0 = time.time()
    p = subprocess.run(['git', 'clone', '--depth', '1', url, str(d)],
                       capture_output=True, text=True, encoding='utf-8', errors='replace',
                       env=env, timeout=600)
    out = ((p.stdout or '') + (p.stderr or '')).strip().splitlines()
    print(f'[{"OK" if p.returncode == 0 else "FAIL"}] {name} ({time.time()-t0:.0f}s) rc={p.returncode}', flush=True)
    for ln in out[-3:]:
        print('   ', ln, flush=True)
print('CLONE_ALL_DONE')
