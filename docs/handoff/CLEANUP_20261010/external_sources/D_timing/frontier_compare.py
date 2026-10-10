# -*- coding: utf-8 -*-
"""四包 + 前沿点对比表 (2026-09-09): 回答"哪些值得提交"""
from __future__ import annotations

import glob
import json
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

print('=== 1.4B 各资产 scale 前沿 (val 3 张) ===')
for pat in ('wave0/gate_m0s1_s4000_sc*.json', 'wave0/gate_m0s7_s5000_sc*.json',
            'wave0/gate_g3w18s1_s4500_sc*.json', 'wave0/gate_g3w20s7_s6500_sc*.json'):
    for p in sorted(glob.glob('D:/tmp_timing/' + pat)):
        d = json.load(open(p, encoding='utf-8'))
        name = p.replace('\\', '/').split('/')[-1]
        print(f'  {name:<45} clip={d["clip"]:.4f} lpips={d["lpips"]:.4f} pass={d["pass"]}')

def fmt(x, nd=4):
    return '  -   ' if x is None else f'{x:.{nd}f}'


print()
print('=== 四包打包资产 (提交对象) ===')
prof = json.load(open('D:/tmp_timing/asset_profiles.json', encoding='utf-8'))
for k, a in prof['assets'].items():
    v = a['val']
    t30 = a['test_nr_30']
    t100 = a.get('test_nr_100')
    c100 = f'{t100["clipiqa"]:.4f}' if t100 else '  -   '
    print(f'  {k:<9} val clip={fmt(v["clipiqa"])} lpips={fmt(v["lpips"])} '
          f'dists={fmt(v["dists"])} musiq={fmt(v["musiq"], 2)} '
          f'| test100 clip={c100} test30 musiq={fmt(t30["musiq"], 2)} '
          f'niqe={fmt(t30["niqe"], 2)}')
