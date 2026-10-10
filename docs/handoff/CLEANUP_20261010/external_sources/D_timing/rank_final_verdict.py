# -*- coding: utf-8 -*-
"""rank 轴最终判定 (2026-09-09 18:25): 配对逐图差异 + 前沿带对比
val 只有 3 张 -> 均值差异必须用配对 per-case 差分判断, 否则被图间方差淹没。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

D = Path('D:/tmp_timing')


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


POINTS = {
    'm0s1_sc0.93 (冠军, r32)': D / 'wave0/gate_m0s1_s4000_sc0.93.json',
    'm0s1_sc0.90 (r32)': D / 'wave0/gate_m0s1_s4000_sc0.90.json',
    'm0s7_sc0.97 (r32)': D / 'wave0/gate_m0s7_s5000_sc0.97.json',
    'r16_s2000_sc0.90': D / 'rank_sc/gate_p512_14b_m0s1r16_s2000_sc0.9.json',
    'r16_s2000_sc0.93': D / 'rank_sc/gate_p512_14b_m0s1r16_s2000_sc0.93.json',
    'r16_s4000_sc0.90': D / 'rank_sc/gate_p512_14b_m0s1r16_s4000_sc0.9.json',
}

print('=== 各点均值 (val 3 张) ===')
for k, p in POINTS.items():
    if not p.exists():
        print(f'  {k:<26} MISSING {p.name}')
        continue
    g = load(p)
    per = g.get('per', {})
    cs = ' | '.join(f'{c}:{per[c]["clipiqa"]:.4f}/{per[c]["lpips"]:.4f}'
                    for c in sorted(per))
    print(f'  {k:<26} clip={g["clip"]:.4f} lpips={g["lpips"]:.4f}  [{cs}]')

print('\n=== 配对逐图差分 (冠军 m0s1_sc0.93 为基准) ===')
base = load(POINTS['m0s1_sc0.93 (冠军, r32)'])['per']
for k, p in POINTS.items():
    if 'm0s1_sc0.93' in k or not p.exists():
        continue
    g = load(p)
    per = g.get('per', {})
    if not per:
        continue
    dl = [per[c]['lpips'] - base[c]['lpips'] for c in sorted(per) if c in base]
    dc = [per[c]['clipiqa'] - base[c]['clipiqa'] for c in sorted(per) if c in base]
    print(f'  {k:<26} Δlpips 逐图 {["%+.4f" % x for x in dl]}  均值 {sum(dl)/len(dl):+.4f}')
    print(f'  {"":<26} Δclip  逐图 {["%+.4f" % x for x in dc]}  均值 {sum(dc)/len(dc):+.4f}')

print('\n=== 判定 ===')
print('  预登记过门 = clip>=0.920 且 lpips<=0.352')
print('  r16 最优 clip>=0.920 点 = s2000-sc0.90 (0.9225, 0.3553) -> lpips 0.3553 > 0.352 => 不过门')
print('  冠军 0.3571 -> r16 好 0.0018 (纯 lpips 公式下 +0.72 分, <1σ) => 无显著收益')
