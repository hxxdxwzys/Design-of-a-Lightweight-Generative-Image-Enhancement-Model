# -*- coding: utf-8 -*-
"""sc0.0 打包完成后 → 速度轴 (2026-09-12 夜)
等 build_M24sc00.json 出现(说明 sc0.0 已打包+门禁完), 然后:
  ① speed_breakdown_512.py  —— 512 时延四组件分解 + DiT 剪枝收益【实测】
  ② verify_probe_a_frugal.py —— 探针 A 本地 niqe 复核 (断点续跑, 省内存)
不重复打包, 不做 sc1.0。
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
MARK = T / 'build_M24sc00.json'
LOG = T / 'chain_after_sc00.log'

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(script, args, timeout, tag, tail=30):
    log(f'--- {tag} ---')
    with (T / f'after_{tag}.log').open('w', encoding='utf-8') as fo:
        r = subprocess.run([PY, '-u', str(T / script)] + args, stdout=fo,
                           stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                           errors='replace', timeout=timeout, cwd=r'C:/models/finals')
    txt = (T / f'after_{tag}.log').read_text(encoding='utf-8', errors='replace')
    for ln in txt.splitlines()[-tail:]:
        log('   | ' + ln)
    return r.returncode


def main():
    log('=== sc0.0 → 速度轴 链启动 (不重复打包, 不做 sc1.0) ===')
    t0 = time.time()
    while not MARK.exists():
        if time.time() - t0 > 90 * 60:
            log('   !! 等 sc0.0 打包超时'); return
        time.sleep(30)
    log('   sc0.0 打包完成: ' + MARK.read_text(encoding='utf-8').replace('\n', ' '))

    # ① 速度轴: 时延分解 + 剪枝收益实测 (必须安静 GPU)
    run('speed_breakdown_512.py', [], 3600, 'speed_breakdown', tail=32)

    # ② 探针 A 本地 niqe 复核
    run('verify_probe_a_frugal.py', ['--n', '30'], 3 * 3600, 'probeA_niqe', tail=14)

    log('=== 链结束; upload/ 现状 ===')
    for p in sorted((MODELS / 'upload').glob('*.zip')):
        log(f'   {p.name}  {p.stat().st_size/2**30:.2f} GiB')
    print('CHAIN_AFTER_SC00_DONE')


if __name__ == '__main__':
    main()
