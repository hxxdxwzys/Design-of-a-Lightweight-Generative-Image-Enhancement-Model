import subprocess, sys, os
from pathlib import Path
env = dict(os.environ, PYTHONIOENCODING='utf-8')
PY = r'D:/Python/python.exe'
for size in ('l', 'xl'):
    print(f'===== pixrestore-{size} =====', flush=True)
    p = subprocess.run([PY, '-u', r'D:/tmp_timing/test_pixrestore.py', '--size', size, '--steps', '1'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, timeout=3600)
    out = ((p.stdout or '') + (p.stderr or ''))
    keep = [l for l in out.splitlines() if any(k in l for k in ('[model]', '[latency]', 'case', '均值', '参照'))]
    print('\n'.join(keep), flush=True)
print('SIZES_DONE')
