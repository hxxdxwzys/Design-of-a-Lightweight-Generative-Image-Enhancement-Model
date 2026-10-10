# -*- coding: utf-8 -*-
"""链 J (排队在 H2 之后): 补"官方口径"盲区 —— val 4K 全图有参四项"""
import subprocess, sys, time
from pathlib import Path
T = Path(r'D:/tmp_timing')
LOG = T / 'chain_20260915j.log'
PY = r'D:/Python/python.exe'


def log(m):
    s = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


t0 = time.time()
while time.time() - t0 < 20 * 3600:
    f = T / 'chain_20260915h2.log'
    if f.exists() and 'CHAIN_H2_DONE' in f.read_text(encoding='utf-8', errors='replace'):
        break
    if time.time() - t0 > 3600 and (time.time() - f.stat().st_mtime) > 2700:
        log('!! H2 疑似中断, 继续 4K 口径测量')
        break
    time.sleep(120)
log('开始 val 4K 有参测量')
try:
    p = subprocess.run([PY, '-u', str(T / 'val4k_fr.py')], capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=7200)
    out = (p.stdout or '') + (p.stderr or '')
except Exception as e:
    out = 'EXC %s' % e
with LOG.open('a', encoding='utf-8') as fh:
    fh.write(out + '\n')
log('CHAIN_J_DONE')
