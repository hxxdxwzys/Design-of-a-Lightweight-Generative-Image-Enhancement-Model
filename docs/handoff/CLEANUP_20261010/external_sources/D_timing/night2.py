# -*- coding: utf-8 -*-
"""重启后串行链 (2026-09-13 18:0x):
  ① 两包门禁 (GPU) ② runner 提速勘察 (GPU) ③ HYPIR 断点续跑 (GPU ~50min) ④ HYPIR 全量细节对比 (CPU)
全部串行, 避免单卡并发; 日志 D:/tmp_timing/night2.log
"""
import os, subprocess, sys, time
from pathlib import Path

PY = r'D:/Python/python.exe'
LOG = Path(r'D:/tmp_timing/night2.log')

def log(msg):
    s = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')

def run(name, cmd, timeout=10800):
    log(f'>>> {name}')
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env, timeout=timeout)
    out = (p.stdout or '') + (p.stderr or '')
    with LOG.open('a', encoding='utf-8') as f:
        f.write(out + '\n')
    log(f'<<< {name} rc={p.returncode}')
    return out

run('① gate ProbeSlow8', [PY, '-u', r'D:/tmp_timing/preupload_gate.py', r'D:/models/upload/Track2_ProbeSlow8.zip'])
run('① gate ProbeDetail', [PY, '-u', r'D:/tmp_timing/preupload_gate.py', r'D:/models/upload/Track2_ProbeDetail.zip'])
run('② speed_probe_v2', [PY, '-u', r'D:/tmp_timing/speed_probe_v2.py'])
run('③ HYPIR 断点续跑', [PY, '-u', r'D:/tmp_timing/hypir_resume.py'])
run('④ HYPIR vs M24 全量', [PY, '-u', r'D:/tmp_timing/hypir_vs_m24.py'])
log('NIGHT2_DONE')
