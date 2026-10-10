# -*- coding: utf-8 -*-
"""HYPIR 结束后自动串行执行: ①HYPIR 全量细节对比 ②两包门禁 ③runner 提速勘察
日志: D:/tmp_timing/night_after_hypir.log
"""
import subprocess, sys, time
from pathlib import Path

PY = r'D:/Python/python.exe'
LOG = Path(r'D:/tmp_timing/night_after_hypir.log')
RES = Path(r'D:/models/hypir/out_test100/result')
HYPIR_LOG = Path(r'D:/tmp_timing/hypir_test100.log')

def log(msg):
    s = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')

def run(name, cmd, timeout=7200):
    log(f'>>> {name}')
    env = dict(__import__('os').environ, PYTHONIOENCODING='utf-8')
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env, timeout=timeout)
    out = (p.stdout or '') + (p.stderr or '')
    with LOG.open('a', encoding='utf-8') as f:
        f.write(out + '\n')
    log(f'<<< {name} rc={p.returncode}')
    return out

# ① 等 HYPIR 出满 100 张
log('等待 HYPIR 100 张完成 ...')
t0 = time.time()
while True:
    n = len(list(RES.glob('*.png'))) if RES.exists() else 0
    if n >= 100:
        log(f'HYPIR 完成 {n}/100, 用时 {(time.time()-t0)/60:.1f} min')
        break
    if time.time() - t0 > 3 * 3600:
        log(f'超时 (只到 {n}/100), 继续后续步骤')
        break
    time.sleep(60)

# ② HYPIR vs M24 细节对比 (全量)
run('hypir_vs_m24 (全量)', [PY, '-u', r'D:/tmp_timing/hypir_vs_m24.py'])

# ③ 门禁
run('gate ProbeSlow8', [PY, '-u', r'D:/tmp_timing/preupload_gate.py', r'D:/models/Track2_ProbeSlow8.zip'])
run('gate ProbeDetail', [PY, '-u', r'D:/tmp_timing/preupload_gate.py', r'D:/models/Track2_ProbeDetail.zip'])

# ④ 提速勘察
run('speed_probe_v2', [PY, '-u', r'D:/tmp_timing/speed_probe_v2.py'])

log('NIGHT_AFTER_HYPIR_DONE')
