# -*- coding: utf-8 -*-
"""探针 A 收尾链 (2026-09-12): 等 100 张 jpg -> 等指标复核 -> QC/remerge/打包/10项门禁/归位 upload/

序列化的理由: QC 与指标复核都要占 GPU, 渲染进程还在时三方争用会互相拖慢甚至 OOM。
用法: python chain_probe_a_finish.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
MODELS = Path(r'D:/models')
OUTDIR = MODELS / 'final_compliant_probeA_sc04' / 'output_dir'
VERIFY = T / 'probe_a_verify.json'
VERIFIER_PID = 19716          # verify_probe_a_metrics.py
LOG = T / 'chain_probe_a_finish.log'

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def alive(pid):
    r = subprocess.run(['powershell', '-NoProfile', '-Command',
                        f'if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{"1"}} else {{"0"}}'],
                       capture_output=True, text=True)
    return r.stdout.strip() == '1'


def wait(cond, timeout_min, what, every=60):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout_min * 60:
            log(f'   [超时] 等待 {what} 超过 {timeout_min} min'); return False
        time.sleep(every)
    log(f'   [ok] {what} ({time.time()-t0:.0f}s)')
    return True


def n_jpg():
    return len(list(OUTDIR.glob('*.jpg')))


def main():
    log('=== 探针 A 收尾链启动 ===')
    if not wait(lambda: n_jpg() >= 100, 180, f'100 张出图 (现 {n_jpg()})'):
        log('   !! 出图不足, 中止'); return
    # 指标复核先跑完 (它是"提交前最后一道证据门"), 再让 QC 独占 GPU
    wait(lambda: VERIFY.exists() or not alive(VERIFIER_PID), 45, '指标复核收尾', every=30)

    log('→ 执行 build_probe_a.py --skip-render (QC/remerge/打包/门禁)')
    t0 = time.time()
    with (T / 'build_probe_a_finish.log').open('w', encoding='utf-8') as fo:
        r = subprocess.run([PY, '-u', str(T / 'build_probe_a.py'), '--scale', '0.4',
                            '--tag', 'probeA_sc04', '--skip-render'],
                           stdout=fo, stderr=subprocess.STDOUT, text=True,
                           encoding='utf-8', errors='replace', cwd=r'C:/models/finals')
    log(f'   rc={r.returncode}, 耗时 {(time.time()-t0)/60:.1f} min')
    for ln in (T / 'build_probe_a_finish.log').read_text(encoding='utf-8', errors='replace').splitlines()[-25:]:
        log('   | ' + ln)
    bj = T / 'build_probeA_sc04.json'
    if bj.exists():
        log('   build json: ' + bj.read_text(encoding='utf-8').replace('\n', ' '))
    log('=== 收尾链结束 ===')
    print('CHAIN_PROBE_A_FINISH_DONE')


if __name__ == '__main__':
    main()
