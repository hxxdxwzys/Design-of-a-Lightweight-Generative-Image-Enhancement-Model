# -*- coding: utf-8 -*-
"""看门: 等 v16 打包完成 -> 掐掉链 M(v17) -> 把卡让给 Q 战场"""
import subprocess, sys, time
from pathlib import Path
LOG = Path(r'D:/tmp_timing/_watch_m.log')
MARK = 'TinyV16'


def log(m):
    s = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + chr(10))


t0 = time.time()
while time.time() - t0 < 3 * 3600:
    out = Path(r'D:/tmp_timing/chain_20260915m.out')
    txt = out.read_text(encoding='utf-8', errors='replace') if out.exists() else ''
    done = ('v16 productize' in txt) or (Path('D:/models/Track2_TinyV16.zip').exists())
    v17 = 'train v17' in txt
    if done or v17:
        log('v16 打包段已到 (done=%s v17=%s) -> 掐链 M' % (done, v17))
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "Where-Object { $_.CommandLine -like '*chain_20260915m*' -or $_.CommandLine -like '*--tag v17*' } | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
        for sh in ('pwsh', 'powershell'):
            try:
                p = subprocess.run([sh, '-NoProfile', '-Command', ps], capture_output=True, text=True, timeout=120)
                if p.returncode == 0:
                    log('killed via %s' % sh)
                    break
            except Exception as e:
                log('%s EXC %s' % (sh, e))
        time.sleep(20)
        try:
            q = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=60)
            log('GPU now %s MiB' % (q.stdout or '').strip())
        except Exception:
            pass
        log('WATCH_M_DONE')
        sys.exit(0)
    time.sleep(20)
log('看门超时')
