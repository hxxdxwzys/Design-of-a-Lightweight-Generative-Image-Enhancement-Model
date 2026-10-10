# -*- coding: utf-8 -*-
"""2026-09-15 链 B: ① PixRestore 各档 val 有参四项 (独立进程, 避免 dinov2 模块冲突)
② 包/M24/文件型配置 ③ HYPIR 测试集出图的三项代理
日志: D:/tmp_timing/chain_20260915b.log
"""
import subprocess, time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'chain_20260915b.log'


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


run('sweep_pix', [PY, '-u', str(T / 'sweep_val_fullref.py'), '--only', 'lq,s1,b1,b10,l1,l10,xl1,xl10'])
run('sweep_pkg', [PY, '-u', str(T / 'sweep_val_fullref.py'),
                  '--only', 'm24,pkgv2,pkgv5,pkgv7,hypir_ct150,hypir_2x1x'])
run('hypir_test30', [PY, '-u', str(T / 'measure_dir.py'), '--dir', 'D:/models/hypir/out_test100/result',
                     '--pattern', 'case{n}.png', '--n', '30', '--metrics', 'niqe,clipiqa,lapvar',
                     '--tag', 'hypir'])
log('CHAIN_B_DONE')
