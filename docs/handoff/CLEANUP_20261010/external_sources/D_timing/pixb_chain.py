# -*- coding: utf-8 -*-
"""PixB 打包链: 等出图完成 -> 质量代理 -> 打包 -> 门禁 -> 归位"""
import os, sys, time, subprocess
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
PKG = Path(r'D:/models/pkg_PixB')
LOG = T / 'pixb_chain.log'


def log(m):
    s = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


def run(name, cmd, timeout=3600):
    log(f'>>> {name}')
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env, timeout=timeout)
    out = (p.stdout or '') + (p.stderr or '')
    with LOG.open('a', encoding='utf-8') as f:
        f.write(out + '\n')
    log(f'<<< {name} rc={p.returncode}')
    return out


# 等 100 张
t0 = time.time()
while time.time() - t0 < 3 * 3600:
    n = len(list((PKG / 'output_dir').glob('*.jpg')))
    if n >= 100:
        break
    time.sleep(60)
n = len(list((PKG / 'output_dir').glob('*.jpg')))
log(f'出图 {n}/100')

run('PixB 质量代理', [PY, '-u', str(T / 'measure_test30.py'), '--dir', str(PKG / 'output_dir')])
run('PixB 时延', [PY, '-u', str(T / 'measure_latency.py'), str(PKG / 'model_dir')])
run('PixB 打包', [PY, '-u', str(T / 'package_generic.py'), '--name', 'Track2_PixB',
                  '--outdir', str(PKG / 'output_dir'), '--modeldir', str(PKG / 'model_dir')])
out = run('PixB 门禁', [PY, '-u', str(T / 'preupload_gate.py'), '--stage-upload',
                        '--upload-name', 'Track2_PixB.zip', r'D:/models/Track2_PixB.zip'])
log('PASS' if 'PREUPLOAD_ALL_PASS' in out else 'FAIL(见日志)')
log('PIXB_CHAIN_DONE')
