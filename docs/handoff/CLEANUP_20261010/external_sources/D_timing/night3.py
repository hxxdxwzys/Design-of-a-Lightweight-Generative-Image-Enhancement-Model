# -*- coding: utf-8 -*-
"""重启后串行链 v3 (2026-09-13 18:1x) —— 作为 DSH 后台任务运行
  ① HYPIR 断点续跑 (~50min)  ② HYPIR vs M24 全量细节对比  ③ runner 提速勘察(已修 bug)
日志: D:/tmp_timing/night3.log
"""
import os, subprocess, time
from pathlib import Path

PY = r'D:/Python/python.exe'
LOG = Path(r'D:/tmp_timing/night3.log')

def log(m):
    s = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')

def run(name, cmd, timeout=10800):
    log(f'>>> {name}')
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env, timeout=timeout)
    with LOG.open('a', encoding='utf-8') as f:
        f.write((p.stdout or '') + (p.stderr or '') + '\n')
    log(f'<<< {name} rc={p.returncode}')
    return p.returncode

run('① HYPIR 断点续跑', [PY, '-u', r'D:/tmp_timing/hypir_resume.py'])
run('② HYPIR vs M24 全量细节', [PY, '-u', r'D:/tmp_timing/hypir_vs_m24.py'])
run('③ runner 提速勘察', [PY, '-u', r'D:/tmp_timing/speed_probe_v2.py'])
log('NIGHT3_DONE')
