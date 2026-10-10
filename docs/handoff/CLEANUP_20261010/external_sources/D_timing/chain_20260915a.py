# -*- coding: utf-8 -*-
"""2026-09-15 链 A: ① val 有参四项扫描 (PixRestore 各档/步数 vs M24) ② HYPIR 出图的测试集代理
顺序执行, 单一 GPU 不并发。日志: D:/tmp_timing/chain_20260915a.log
"""
import subprocess, time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'chain_20260915a.log'


def log(m):
    s = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


def run(name, cmd, timeout=7200):
    log('>>> ' + name)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    out = (p.stdout or '') + (p.stderr or '')
    with LOG.open('a', encoding='utf-8') as f:
        f.write(out + '\n')
    log('<<< %s rc=%d (%.1f min)' % (name, p.returncode, (time.time() - t0) / 60))
    return p.returncode, out


run('sweep_val_fullref', [PY, '-u', str(T / 'sweep_val_fullref.py')])
run('hypir_test30', [PY, '-u', str(T / 'measure_test30.py'), '--dir', 'D:/models/hypir/out_test100/result', '--n', '30'])
log('CHAIN_A_DONE')
