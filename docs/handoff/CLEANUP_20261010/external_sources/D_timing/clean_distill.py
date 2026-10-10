# -*- coding: utf-8 -*-
"""清理蒸馏数据: 删掉早期 256 域的旧对 (混尺寸导致 np.stack 崩)"""
from pathlib import Path
from PIL import Image
LQ = Path(r'D:/datasets/distill256/lq'); TGT = Path(r'D:/datasets/distill256/tgt')
bad = []
for p in sorted(LQ.glob('*.jpg')):
    with Image.open(p) as im:
        if im.size != (512, 512):
            bad.append(p)
for p in bad:
    q = TGT / p.name
    p.unlink()
    if q.exists():
        q.unlink()
print(f'删除 {len(bad)} 对非 512 的旧数据')
n = len(list(LQ.glob('*.jpg')))
print(f'剩下 {n} 对')
# 尺寸一致性复核
sizes = set()
for p in sorted(LQ.glob('*.jpg'))[:200]:
    with Image.open(p) as im:
        sizes.add(im.size)
print('抽样尺寸:', sizes)
print('CLEAN_DISTILL_DONE')
