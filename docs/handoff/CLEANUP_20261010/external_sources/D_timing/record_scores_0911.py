# -*- coding: utf-8 -*-
"""9-11 分数入库 + 公式反演 (2026-09-11 晚)
数据: 用户提供的 9 发官方分 (按提交时间归因)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

P = Path(r'D:/tmp_timing/asset_profiles.json')
d = json.loads(P.read_text(encoding='utf-8'))

# 提交时间 -> 官方分 (用户 2026-09-11 提供)
SCORES = {
    '2026-09-11 16:32:41': 31.8185,   # G3W18S1_v3
    '2026-09-11 16:26:32': 34.1056,   # M0S1_v3
    '2026-09-11 16:21:17': 30.6253,   # M0S7_SC10_v3
    '2026-09-11 16:16:14': 30.9363,   # M0S7_SC097_v3
    '2026-09-11 16:11:27': 43.6951,   # M24_v3
    '2026-09-11 15:31:59': 34.0989,   # M0S1_v2
    '2026-09-11 15:06:58': 30.5914,   # M0S7_SC10_v2
    '2026-09-11 15:01:03': 31.3942,   # M0S7_SC097 (v2 版)
    '2026-09-11 14:56:47': 43.6731,   # M24_v2
}
for s in d['submissions']:
    if s['time'] in SCORES:
        s['official'] = SCORES[s['time']]
        s['status'] = '测评完成'

# 按资产汇总 (v2/v3 取均值, 并记录配对的差值 = 官方确定性)
by_asset = {}
for s in d['submissions']:
    if s.get('official') is None:
        continue
    by_asset.setdefault(s['asset'], []).append((s['pack'], s['official']))

print('=== 各资产官方分 (v2/v3 配对) ===')
summary = {}
for a, lst in by_asset.items():
    vals = [v for _, v in lst]
    mean = sum(vals) / len(vals)
    delta = (max(vals) - min(vals)) if len(vals) > 1 else None
    summary[a] = mean
    ds = f'  配对差 {delta:.4f}' if delta is not None else ''
    print(f"  {a:<12} {' + '.join(f'{p}={v:.4f}' for p, v in lst)}  -> 均值 {mean:.4f}{ds}")

print()
print('=== 关键 Δ ===')
def g(a):
    return summary.get(a)

core = g('m0s1') - g('M24')
print(f'  Δ(m0s1 − M24)      = {core:+.4f}')
print(f'     预登记两支: +9.8 (公式认 lpips, 双方 clip 饱和) | −20.5 (clip 未饱和段主导)')
print(f'  Δ(m0s7_sc10 − m0s1) = {g("m0s7_sc10") - g("m0s1"):+.4f}  (两者 val 画像近同位)')
print(f'  Δ(g3w18s1 − m0s1)   = {g("g3w18s1") - g("m0s1"):+.4f}  (dists 优势点)')
print(f'  Δ(m0s7_s097 − m0s1) = {g("m0s7_s097") - g("m0s1"):+.4f}  (两轴支配 m0s1)')

d['_official_scores_20260911'] = {
    '_doc': '9-11 九发官方分 (用户提供); v2/v3 是同一资产同一批 output_dir 图像 -> 配对差即官方确定性',
    'by_asset_mean': {k: round(v, 4) for k, v in summary.items()},
    'pair_delta': {a: round(max(v for _, v in lst) - min(v for _, v in lst), 4)
                   for a, lst in by_asset.items() if len(lst) > 1},
    'key_deltas': {'m0s1_minus_M24': round(core, 4),
                   'm0s7sc10_minus_m0s1': round(g('m0s7_sc10') - g('m0s1'), 4),
                   'g3w18s1_minus_m0s1': round(g('g3w18s1') - g('m0s1'), 4),
                   'm0s7s097_minus_m0s1': round(g('m0s7_s097') - g('m0s1'), 4)},
    'best': max(summary.items(), key=lambda kv: kv[1]),
}
d['official_scores']['M24'] = round(g('M24'), 4)
d['official_scores']['m0s1'] = round(g('m0s1'), 4)
d['official_scores']['g3w18s1'] = round(g('g3w18s1'), 4)
d['official_scores']['m0s7'] = round(g('m0s7_sc10'), 4)
P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('\n已入库 asset_profiles.json')
