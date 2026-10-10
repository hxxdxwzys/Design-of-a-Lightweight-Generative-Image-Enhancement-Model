# -*- coding: utf-8 -*-
"""等四包指标测量结束后跑 HYPIR 100 张 (单 GPU 重活纪律, 不并发)"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'chain_hypir.log'


def busy():
    r = subprocess.run(['powershell', '-NoProfile', '-Command',
                        "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                        "Where-Object { $_.CommandLine -match 'measure_four_pkgs' }).Count"],
                       capture_output=True, text=True)
    try:
        return int((r.stdout or '0').strip()) > 0
    except Exception:
        return False


def log(m):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    log('=== 等待四包测量结束 ===')
    t0 = time.time()
    while busy():
        if time.time() - t0 > 180 * 60:
            log('   !! 超时, 仍继续'); break
        time.sleep(30)
    log('   已空闲, 启动 HYPIR test100')
    with (T / 'after_hypir.log').open('w', encoding='utf-8') as fo:
        subprocess.run([PY, '-u', str(T / 'hypir_test100.py')], stdout=fo,
                       stderr=subprocess.STDOUT, text=True, cwd=r'C:/models/finals')
    for ln in (T / 'after_hypir.log').read_text(encoding='utf-8', errors='replace').splitlines()[-12:]:
        log('   | ' + ln)
    print('CHAIN_HYPIR_DONE')


if __name__ == '__main__':
    main()
