# -*- coding: utf-8 -*-
"""2026-09-15 链 C: ① PixRestore 慢配置 (自适应时延) ② 包/文件型配置 ③ HYPIR test30 ④ v7gan 训练轨迹扫描
日志: D:/tmp_timing/chain_20260915c.log
"""
import subprocess, time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'chain_20260915c.log'


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


SW = str(T / 'sweep_val_fullref.py')
run('sweep_pix_slow', [PY, '-u', SW, '--only', 'b10,l1,l10,xl1,xl10'])
run('sweep_pkg', [PY, '-u', SW, '--only', 'pkgv2,pkgv5,pkgv7,hypir_ct150,hypir_2x1x'])
run('hypir_test30', [PY, '-u', str(T / 'measure_dir.py'), '--dir', 'D:/models/hypir/out_test100/result',
                     '--pattern', 'case{n}.png', '--n', '30', '--metrics', 'niqe,clipiqa,lapvar', '--tag', 'hypir'])
for tag, ck in [('v7g8k', 'v7gan_s8000.pt'), ('v7g16k', 'v7gan_s16000.pt'), ('v7g25k', 'v7gan_s25000.pt')]:
    d = 'D:/models/traj/' + tag
    run('render_' + tag, [PY, '-u', str(T / 'render_subset.py'), '--ckpt', 'D:/models/tiny_v1/' + ck,
                          '--out', d, '--down', '2', '--n', '30', '--order', 'lex'])
    run('measure_' + tag, [PY, '-u', str(T / 'measure_dir.py'), '--dir', d, '--n', '30',
                           '--metrics', 'niqe,clipiqa,lapvar', '--tag', tag])
log('CHAIN_C_DONE')
