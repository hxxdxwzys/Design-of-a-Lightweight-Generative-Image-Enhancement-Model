# -*- coding: utf-8 -*-
"""rank 轴门控汇总 (2026-09-09 18:00): r8/r16 全档 vs 冠军 m0s1
判定: 是否存在 (clip >= 0.919 且 lpips < 0.3571) 的点 = 地板带被打破
"""
from __future__ import annotations

import glob
import json
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CHAMP = (0.9261, 0.3571)   # m0s1-s4000-sc0.93 (r32) val
GATE_CLIP, GATE_LPIPS = 0.919, 0.360

rows = {}
for pat in ('gate_p512_14b_m0s1r8_lora_s*.json', 'gate_p512_14b_m0s1r16_lora_s*.json',
            'gate_p512_14b_m0s1_lora_s*.json'):
    for p in sorted(glob.glob('D:/tmp_timing/p100_val/' + pat)):
        d = json.load(open(p, encoding='utf-8'))
        name = p.replace('\\', '/').split('/')[-1]
        var = name.replace('gate_', '').replace('_lora_s', '|').replace('.json', '')
        rows.setdefault(var.split('|')[0], []).append((int(var.split('|')[1]), d['clip'], d['lpips']))

for var in sorted(rows):
    pts = sorted(rows[var])
    print(f'=== {var} ===')
    for step, clip, lpips in pts:
        flag = ''
        if clip >= GATE_CLIP and lpips < CHAMP[1]:
            flag = '  ⭐ 破地板带!'
        elif clip >= CHAMP[0] and lpips < CHAMP[1]:
            flag = '  ⭐ 同 clip 更高保真'
        print(f'  s{step:<5} clip={clip:.4f} lpips={lpips:.4f}{flag}')
    # 前沿: clip>=0.919 内最低 lpips
    ok = [x for x in pts if x[1] >= GATE_CLIP]
    if ok:
        best = min(ok, key=lambda x: x[2])
        print(f'  → clip≥{GATE_CLIP} 内最低 lpips: s{best[0]} ({best[1]:.4f}, {best[2]:.4f}) '
              f'vs 冠军 {CHAMP[1]:.4f} = {best[2]-CHAMP[1]:+.4f}')
    print()

print(f'冠军 m0s1-s4000-sc0.93 (r32): clip={CHAMP[0]} lpips={CHAMP[1]}')
