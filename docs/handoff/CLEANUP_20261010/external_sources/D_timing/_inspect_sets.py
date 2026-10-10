# -*- coding: utf-8 -*-
import sys, os, zipfile, zlib
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
names = sorted(TEST.glob('*.jpg'))
print('[test] n=%d first5=%s' % (len(names), [f.name for f in names[:5]]))
idx = [''.join(c for c in f.stem if c.isdigit()) for f in names[:30]]

dirs = ['final_compliant','pkg_M24img_V9brun','pkg_PixB','pkg_TinyV1','pkg_TinyV2','pkg_TinyV2g',
        'pkg_TinyV4g','pkg_TinyV5d','pkg_TinyV7gan','pkg_TinyV9b','probe_a','probe_native','probe_noise',
        'wave2_m0s1_s090','wave2_m0s1_s097','wave2_m0s1_s100','wave2_m0s7_s097']
for d in dirs:
    p = Path(r'D:/models') / d
    if not p.exists():
        print('%-22s MISSING' % d); continue
    js = sorted(p.glob('*.jpg'))
    fs = sorted(p.glob('*'))
    hit = sum(1 for i in idx if (p / ('case%s.jpg' % i)).exists())
    print('%-22s n=%3d files=%3d caseN_cov=%2d/30 first=%s' % (d, len(js), len(fs), hit, fs[0].name if fs else '-'))

# map wave2 dirs + probe_a to official score via CRC of first image inside the submitted zip
HIST = Path(r'C:/生成式图像增强轻量模型设计挑战/提交历史')
zips = {}
for z in sorted(HIST.glob('*.zip')):
    try:
        with zipfile.ZipFile(z) as zf:
            ent = [e for e in zf.infolist() if e.filename.startswith('output_dir/') and e.filename.lower().endswith('.jpg')]
            ent.sort(key=lambda e: e.filename)
            if ent:
                zips[z.name] = (len(ent), ent[0].filename, ent[0].CRC)
    except Exception as e:
        print('!! zip %s %s' % (z.name, e))
print('--- zip first-image CRC ---')
for k, v in zips.items():
    print('%-42s n=%3d %s crc=%08x' % (k, v[0], v[1], v[2]))
print('--- local dir first-image CRC (case-index order) ---')
for d in dirs:
    p = Path(r'D:/models') / d
    if not p.exists(): continue
    f = p / ('case%s.jpg' % idx[0])
    if f.exists():
        print('%-22s %s crc=%08x size=%d' % (d, f.name, zlib.crc32(f.read_bytes()) & 0xffffffff, f.stat().st_size))
