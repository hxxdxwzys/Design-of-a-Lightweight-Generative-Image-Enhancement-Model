# -*- coding: utf-8 -*-
"""把 val->test clipiqa 转移写入 asset_profiles.json (2026-09-10)"""
from __future__ import annotations

import json
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

P = 'D:/tmp_timing/asset_profiles.json'
d = json.load(open(P, encoding='utf-8'))
TEST = {'M24': (0.9008, 100), 'm0s1': (0.8780, 100),
        'g3w18s1': (0.8612, 30), 'm0s7': (0.8790, 30)}
for k, a in d['assets'].items():
    tc, n = TEST[k]
    a['val_to_test_clip'] = {'test_clip': tc, 'drop': round(a['val']['clipiqa'] - tc, 4),
                             'n_test': n}
json.dump(d, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

print('=== val -> test clipiqa 转移（1.4B 系统性大跌 = 关键风险）===')
for k, a in d['assets'].items():
    r = a['val_to_test_clip']
    print('  {:<9} val={:.4f} -> test={:.4f}  跌幅 {:.4f} (n={})'.format(
        k, a['val']['clipiqa'], r['test_clip'], r['drop'], r['n_test']))
print()
print('=== 第①→② 判别的两支预期（Δ = score(m0s1) − score(M24)）===')
print('  公式 clip 饱和线 T >= 0.9008（双方均饱和）-> 只比 lpips -> Δ ≈ +9.8  => 1.4B 上位')
print('  公式 clip 饱和线 T <  0.8780（仅 M24 饱和）-> clip 差 0.0228 计入 -> Δ ≈ −20.5 => M24 上位')
print('  （旧世界实测 T ∈ (0.872, 0.905)，正落在两支之间 -> 这一发直接钉死 T）')
