# -*- coding: utf-8 -*-
"""按用户 2026-09-11 实际提交列表更新台账 (v2 四发 + v3 五发 = 9 发在评测)"""
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

SUB = [
    # 9-10 第一批 (四发, 官方报错 "Failed to load IQAEvaluator", 不计次数)
    {'n': 1, 'id': '1789040888498', 'time': '2026-09-10 19:50:33', 'asset': 'M24',
     'file': 'Track2_CompliantM24.zip', 'pack': 'v1(未修)', 'status': '质量评测失败(不计次数)'},
    {'n': 2, 'id': '1789041713429', 'time': '2026-09-10 20:05:53', 'asset': 'm0s7_sc10',
     'file': 'Track2_14B_M0S7_SC10.zip', 'pack': 'v1(未修)', 'status': '质量评测失败(不计次数)'},
    {'n': 3, 'id': '1789041984124', 'time': '2026-09-10 20:10:25', 'asset': 'm0s1',
     'file': 'Track2_14B_M0S1_SC093.zip', 'pack': 'v1(未修)', 'status': '质量评测失败(不计次数)'},
    {'n': 4, 'id': '1789042263914', 'time': '2026-09-10 20:15:04', 'asset': 'g3w18s1',
     'file': 'Track2_14B_G3W18S1_SC093.zip', 'pack': 'v1(未修)', 'status': '质量评测失败(不计次数)'},
    # 9-11 第一批 (用户在我加固前交的 v2 四发)
    {'n': 5, 'id': '1789109688993', 'time': '2026-09-11 14:56:47', 'asset': 'M24',
     'file': 'Track2_M24_v2.zip', 'pack': 'v2', 'status': '测评中'},
    {'n': 6, 'id': '1789109825095', 'time': '2026-09-11 15:01:03', 'asset': 'm0s7_s097',
     'file': 'Track2_M0S7_SC097.zip', 'pack': 'v2(波2首版)', 'status': '待评测'},
    {'n': 7, 'id': '1789110179634', 'time': '2026-09-11 15:06:58', 'asset': 'm0s7_sc10',
     'file': 'Track2_M0S7_SC10_v2.zip', 'pack': 'v2', 'status': '待评测'},
    {'n': 8, 'id': '1789111677996', 'time': '2026-09-11 15:31:59', 'asset': 'm0s1',
     'file': 'Track2_M0S1_SC093_v2.zip', 'pack': 'v2', 'status': '待评测'},
    # 9-11 第二批 (加固后的 v3 五发)
    {'n': 9, 'id': '1789114168904', 'time': '2026-09-11 16:11:27', 'asset': 'M24',
     'file': 'Track2_M24_v3.zip', 'pack': 'v3(加固)', 'status': '待评测'},
    {'n': 10, 'id': '1789114336591', 'time': '2026-09-11 16:16:14', 'asset': 'm0s7_s097',
     'file': 'Track2_M0S7_SC097_v3.zip', 'pack': 'v3(加固)', 'status': '待评测'},
    {'n': 11, 'id': '1789114639363', 'time': '2026-09-11 16:21:17', 'asset': 'm0s7_sc10',
     'file': 'Track2_M0S7_SC10_v3.zip', 'pack': 'v3(加固)', 'status': '待评测'},
    {'n': 12, 'id': '1789114950519', 'time': '2026-09-11 16:26:32', 'asset': 'm0s1',
     'file': 'Track2_M0S1_SC093_v3.zip', 'pack': 'v3(加固)', 'status': '待评测'},
    {'n': 13, 'id': '1789115322839', 'time': '2026-09-11 16:32:41', 'asset': 'g3w18s1',
     'file': 'Track2_G3W18S1_SC093_v3.zip', 'pack': 'v3(加固)', 'status': '待评测'},
]
d['submissions'] = SUB
d['budget'].update({
    'quota': 15,
    'used_new_world': 9,
    'remaining': 6,
    'note_9_11': '用户 2026-09-11 先交了 v2 四发(旧结构), 后又交 v3 五发(加固结构) = 9 发在评测; '
                 '若 v2 四发也成功计次, 则剩 6 发(用户认为可接受); 若 v2 失败则退回次数',
})

d['_v2_v3_pairing'] = {
    '_doc': '同一资产的 v2/v3 两发: output_dir 的 100 张 jpg 完全相同, 仅 model_dir 结构不同',
    'implication': '若两发都成功 -> 质量分应【完全相等】(质量只看 output_dir 图像, 确定性指标) '
                   '=> 可当官方评测确定性检验; 若两发表现不同 -> 说明质量评测受 model_dir 影响(异常)',
    'diagnostic': '若 v2 四发失败而 v3 五发成功 => 证明"官方 harness 会再插一次 model_dir"的对抗场景真实存在, '
                  '_vendor 加固是必需的(否则会重演 9-10)',
    'pairs': {
        'M24': ('1789109688993 (v2)', '1789114168904 (v3)'),
        'm0s7_s097': ('1789109825095 (v2)', '1789114336591 (v3)'),
        'm0s7_sc10': ('1789110179634 (v2)', '1789114639363 (v3)'),
        'm0s1': ('1789111677996 (v2)', '1789114950519 (v3)'),
        'g3w18s1': ('—', '1789115322839 (v3)'),
    },
}

d['_delta_plan_20260911'] = {
    '_doc': '拿到分数后的判读映射 (v2/v3 取同一资产的分, 若两发都有则先比是否相等)',
    'core': {'Δ(m0s1 − M24)': '≈ +9.8 -> 公式认 lpips (1.4B 上位) | ≈ −20.5 -> clip/无参主导 (M24 终局)'},
    'sigma': {'Δ(m0s7_sc10 − m0s1)': '两资产近同位 -> 该 Δ 即官方噪声 σ 的估计'},
    'dists': {'Δ(g3w18s1 − m0s1)': '破衡 w_dists = −84.1'},
    'wave2': {'Δ(m0s7_s097 − m0s1)': 'm0s7_s097 在 (test clip 0.8838, val lpips 0.3545) 两轴支配 m0s1 '
                                     '-> 若 Δ ≤ 0 说明公式有异常交互'},
}

P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('台账已更新:')
print(f"  提交总数 {len(SUB)} | 计次 9 | 余 {d['budget']['remaining']}")
for s in SUB:
    print(f"  #{s['n']:<2} {s['time']}  {s['pack']:<12} {s['file']:<40} {s['status']}")
