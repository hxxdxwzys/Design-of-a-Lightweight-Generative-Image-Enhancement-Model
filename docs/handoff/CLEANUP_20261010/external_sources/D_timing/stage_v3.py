# -*- coding: utf-8 -*-
"""把 5 个 v3 加固包归位到 upload/, 删除被取代的 v2 (2026-09-11)"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

UP = Path(r'D:/models/upload')
UP.mkdir(exist_ok=True)
V3 = ['Track2_M24_v3.zip', 'Track2_M0S1_SC093_v3.zip', 'Track2_M0S7_SC10_v3.zip',
      'Track2_G3W18S1_SC093_v3.zip', 'Track2_M0S7_SC097_v3.zip']

print('=== 归位 v3 (硬链接, 零额外磁盘) ===')
for n in V3:
    src, dst = Path(r'D:/models') / n, UP / n
    if not src.exists():
        print(f'  缺 {n}'); continue
    if dst.exists():
        dst.unlink()
    os.link(src, dst)
    print(f'  OK  {n}  {dst.stat().st_size/2**30:.2f} GiB')

print('\n=== 删除被取代的 v2 ===')
for f in sorted(UP.glob('*.zip')):
    if '_v3' in f.name:
        continue
    sz = f.stat().st_size / 2**30
    f.unlink()
    print(f'  删除 {f.name}  ({sz:.2f} GiB)')

print('\n=== upload/ 最终清单（唯一上传位置）===')
for f in sorted(UP.glob('*.zip')):
    print(f'  {f.stat().st_size/2**30:.2f} GiB  {f.name}')

print('\n=== runner 行结构核验（空行是否本来就多）===')
for tag, zp in (('v3', r'D:/models/Track2_M24_v3.zip'),):
    b = zipfile.ZipFile(zp).read('model_dir/runner.py')
    L = b.split(b'\n')
    print(f'  {tag} runner: 总行 {len(L)}, 非空行 {sum(1 for x in L if x.strip())}, '
          f'CRLF {b.count(chr(13).encode()+chr(10).encode())}')
