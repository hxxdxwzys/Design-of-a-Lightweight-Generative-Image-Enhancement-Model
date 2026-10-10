# -*- coding: utf-8 -*-
"""等 night2 链结束 -> 跑修好的 runner 提速勘察 (避免与 HYPIR 抢显卡)"""
import subprocess, time
from pathlib import Path
PY = r'D:/Python/python.exe'
LOG = Path(r'D:/tmp_timing/night2.log')
OUT = Path(r'D:/tmp_timing/post_night2.log')

def log(m):
    s = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(s, flush=True)
    with OUT.open('a', encoding='utf-8') as f:
        f.write(s + '\n')

log('等待 night2 结束 ...')
t0 = time.time()
while time.time() - t0 < 4 * 3600:
    try:
        if 'NIGHT2_DONE' in LOG.read_text(encoding='utf-8', errors='replace'):
            break
    except Exception:
        pass
    time.sleep(60)
log('night2 结束 (或超时), 开始提速勘察')
env = dict(__import__('os').environ, PYTHONIOENCODING='utf-8')
p = subprocess.run([PY, '-u', r'D:/tmp_timing/speed_probe_v2.py'], capture_output=True,
                   text=True, encoding='utf-8', errors='replace', env=env, timeout=3600)
with OUT.open('a', encoding='utf-8') as f:
    f.write((p.stdout or '') + (p.stderr or '') + '\n')
log(f'speed_probe_v2 rc={p.returncode}')
log('POST_NIGHT2_DONE')
