# -*- coding: utf-8 -*-
"""等 M24p4 打包完成后自动跑 2 样本慢速探针 (2026-09-13)"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
MARK = T / 'build_M24p4.json'
LOG = T / 'chain_after_prune.log'

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
    log('=== 等待 M24p4 打包 ===')
    t0 = time.time()
    while not MARK.exists():
        if time.time() - t0 > 150 * 60:
            log('   !! 超时, 仍继续'); break
        time.sleep(60)
    if MARK.exists():
        log('   M24p4 完成: ' + MARK.read_text(encoding='utf-8').replace('\n', ' '))
    log('=== 启动 2 样本慢速探针 ===')
    with (T / 'after_prune_avg2.log').open('w', encoding='utf-8') as fo:
        subprocess.run([PY, '-u', str(T / 'chain_avg2.py'), '--tag', 'M24avg2'],
                       stdout=fo, stderr=subprocess.STDOUT, text=True,
                       cwd=r'C:/models/finals')
    for ln in (T / 'after_prune_avg2.log').read_text(encoding='utf-8',
                                                     errors='replace').splitlines()[-12:]:
        log('   | ' + ln)
    log('=== upload/ 现状 ===')
    for p in sorted((Path(r'D:/models/upload')).glob('*.zip')):
        log(f'   {p.name}  {p.stat().st_size/2**30:.2f} GiB')
    print('CHAIN_AFTER_PRUNE_DONE')


if __name__ == '__main__':
    main()
