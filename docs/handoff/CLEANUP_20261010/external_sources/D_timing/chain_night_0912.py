# -*- coding: utf-8 -*-
"""夜间总链 (2026-09-12):
  ① 等探针 A 归位 upload/
  ② 【免费验证 1】多种子平均在验证集上的真实增益 (决定阶段 2 值不值得花 12.7h + 1 发)
  ③ 【阶段 1】渲染并打包 sc0.0 (保真极点) + sc1.0 (感知极点)
用法: python chain_night_0912.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
MODELS = Path(r'D:/models')
UPLOAD = MODELS / 'upload'
LOG = T / 'chain_night_0912.log'

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def wait_zip(timeout_min=240):
    t0 = time.time()
    while not (UPLOAD / 'Track2_probeA_sc04.zip').exists():
        if time.time() - t0 > timeout_min * 60:
            log('   !! 等探针 A 超时'); return False
        time.sleep(60)
    return True


def run(script, args, timeout, tag):
    log(f'--- {tag} ---')
    with (T / f'night_{tag}.log').open('w', encoding='utf-8') as fo:
        r = subprocess.run([PY, '-u', str(T / script)] + args, stdout=fo,
                           stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                           errors='replace', timeout=timeout, cwd=r'C:/models/finals')
    txt = (T / f'night_{tag}.log').read_text(encoding='utf-8', errors='replace')
    for ln in txt.splitlines()[-16:]:
        log('   | ' + ln)
    return r.returncode


def main():
    log('=== 夜间总链启动 ===')
    if not wait_zip():
        return
    log('   探针 A 已归位 ✓')

    # ② 免费验证 1: 多种子平均 (4 种子 x 3 张 4K ≈ 15 min)
    run('test_seed_ensemble.py', ['--seeds', '8,108,208,308', '--scale', '0.8'], 3600, 'seed_ens')

    # ③ 阶段 1: sc0.0 + sc1.0
    run('run_stage1_scale_poles.py', [], 12 * 3600, 'stage1')

    log('=== 夜间总链结束; upload/ 现状 ===')
    for p in sorted(UPLOAD.glob('*.zip')):
        log(f'   {p.name}  {p.stat().st_size/2**30:.2f} GiB')
    print('CHAIN_NIGHT_DONE')


if __name__ == '__main__':
    main()
