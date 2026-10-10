# -*- coding: utf-8 -*-
import sys, os, zipfile, zlib, json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
M = Path(r'D:/models')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
names = sorted(TEST.glob('*.jpg'))[:30]
idx = [''.join(c for c in f.stem if c.isdigit()) for f in names]

zips = {}
HIST = Path(r'C:/生成式图像增强轻量模型设计挑战/提交历史')
for z in sorted(HIST.glob('*.zip')):
    try:
        with zipfile.ZipFile(z) as zf:
            e = [x for x in zf.infolist() if x.filename.startswith('output_dir/') and x.filename.lower().endswith('.jpg')]
            if e:
                zips[('%08x' % (e[0].CRC & 0xffffffff))] = '%s  score=%s' % (z.name, z.name.split('——')[-1].replace('.zip',''))
    except Exception as ex:
        print('!!', z.name, ex)

print('=== dirs: resolve output image dir ===')
cands = []
for d in sorted(M.iterdir()):
    if not d.is_dir(): continue
    for sub in ('output_dir', ''):
        p = d / sub if sub else d
        if p.is_dir() and len(list(p.glob('case*.jpg'))) >= 50:
            f = p / ('case%s.jpg' % idx[0])
            if f.exists():
                c = '%08x' % (zlib.crc32(f.read_bytes()) & 0xffffffff)
                cands.append((d.name, sub or '.', len(list(p.glob('case*.jpg'))), c, zips.get(c, 'UNSCORED')))
            break
for n, sub, cnt, c, s in cands:
    print('%-24s sub=%-11s n=%3d crc=%s  %s' % (n, sub, cnt, c, s))
print()
print('=== wave2 / probe_a layout ===')
for d in ['wave2_m0s1_s090','wave2_m0s1_s097','wave2_m0s1_s100','wave2_m0s7_s097','probe_a','probe_noise','final_compliant','pkg_M24img_V9brun']:
    p = M / d
    if not p.exists(): print(d, 'MISSING'); continue
    print('---', d)
    for f in sorted(p.iterdir()):
        print('    ', f.name, '<DIR>' if f.is_dir() else f.stat().st_size)
