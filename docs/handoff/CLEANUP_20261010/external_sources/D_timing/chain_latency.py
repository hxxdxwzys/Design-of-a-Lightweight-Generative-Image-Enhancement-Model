# -*- coding: utf-8 -*-
"""等 avg2 包完成后 -> 实测三个包的真实 512 时延 (安静 GPU)"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
MARK = T / 'build_M24avg2.json'
LOG = T / 'chain_latency.log'

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    log('=== 等 M24avg2 打包 ===')
    t0 = time.time()
    while not MARK.exists():
        if time.time() - t0 > 300 * 60:
            log('   !! 超时, 仍继续'); break
        time.sleep(60)
    if MARK.exists():
        log('   M24avg2 完成: ' + MARK.read_text(encoding='utf-8').replace('\n', ' '))
    time.sleep(20)
    log('=== 实测三个包的 512 时延 (GPU 已安静) ===')
    with (T / 'after_latency.log').open('w', encoding='utf-8') as fo:
        subprocess.run([PY, '-u', str(T / 'measure_pkg_latency.py')],
                       stdout=fo, stderr=subprocess.STDOUT, text=True, cwd=r'C:/models/finals')
    for ln in (T / 'after_latency.log').read_text(encoding='utf-8', errors='replace').splitlines()[-10:]:
        log('   | ' + ln)
    log('=== upload/ 最终现状 ===')
    for p in sorted((Path(r'D:/models/upload')).glob('*.zip')):
        log(f'   {p.name}  {p.stat().st_size/2**30:.2f} GiB')
    print('CHAIN_LATENCY_DONE')


if __name__ == '__main__':
    main()
