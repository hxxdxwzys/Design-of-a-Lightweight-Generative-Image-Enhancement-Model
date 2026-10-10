# -*- coding: utf-8 -*-
"""失败不计次 -> 台账更新 + 第 2 批(5 发)提交计划 (2026-09-11)"""
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

# ① 四发失败 = 不计次 -> 状态补注
for s in d['submissions']:
    s['status'] = '质量评测失败(quality stage) — 官方不计次数, 额度已返还'
    s['quota_charged'] = False

# ② 预算刷新
d['budget']['total_note'] = ('用户 2026-09-11 确认: 评测失败的提交【不占用次数】 -> 9/10 那四发不计, '
                             '额度仍为 15; 本批计划交 5 发, 交完应剩 10')
d['budget']['quota'] = 15
d['budget']['used_new_world'] = 0
d['budget']['remaining'] = 15
d['budget']['refund_note'] = '4 发质量评测失败(根因 A: runner 桩包遮蔽 pyiqa) — 官方不计次数'

# ③ 第 2 批提交计划
d['batch2_plan'] = {
    '_doc': '2026-09-11 第 2 批: 修复版四发 + 波2首选, 全部已过 preupload_gate 8/8',
    'dir': 'D:/models/upload/',
    'order': [
        {'n': 1, 'file': 'Track2_M24_v2_Yuki_18851225671.zip', 'asset': 'M24',
         'role': '校准弹 (新世界第 1 个方程, 0.5B 基准)'},
        {'n': 2, 'file': 'Track2_M0S7_SC10_v2_Yuki_18851225671.zip', 'asset': 'm0s7',
         'role': 'σ 测量 (与 m0s1 近同位, Δ≈0 -> Δ 即官方噪声)'},
        {'n': 3, 'file': 'Track2_M0S1_SC093_v2_Yuki_18851225671.zip', 'asset': 'm0s1',
         'role': 'lpips 主轴判别 (核心): Δ(③-①) ≈ +9.8 -> 1.4B 上位 / ≈ -20.5 -> M24 终局'},
        {'n': 4, 'file': 'Track2_G3W18S1_SC093_v2_Yuki_18851225671.zip', 'asset': 'g3w18s1',
         'role': 'dists 判别 (破衡 w_dists = -84.1)'},
        {'n': 5, 'file': 'Track2_M0S7_SC097_Yuki_18851225671.zip', 'asset': 'm0s7_s097',
         'role': '波2首选 (test clip 0.8838 第一, val lpips 0.3545; 两轴支配其它 1.4B)'},
    ],
    'after': '交完把每发的 提交ID/时间 + 官方分 发我 -> discriminate.py --record 跑 Δ 判读; 应剩 10 发',
}

P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')

print('=== 台账已更新 ===')
print('  额度:', d['budget']['quota'], '| 已计次:', d['budget']['used_new_world'],
      '| 剩余:', d['budget']['remaining'])
print('  四发失败已标: 不计次数')
print()
print('=== 第 2 批上传清单 (D:/models/upload/) ===')
for it in d['batch2_plan']['order']:
    print(f"  {it['n']}. {it['file']}")
    print(f"     {it['role']}")
