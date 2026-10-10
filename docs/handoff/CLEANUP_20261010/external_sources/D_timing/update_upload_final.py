# -*- coding: utf-8 -*-
"""台账更新: 最终上传清单改为 v3 无后缀命名 (2026-09-11, 用户决定)"""
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

d['_upload_ready'] = {
    '_doc': '唯一上传位置 = D:/models/upload/ ; v3 = 结构加固版(桩包移入 model_dir/_vendor/, '
            '标准+对抗两种 IQA 遮蔽审计均 PASS), 5 包全部 PREUPLOAD_GATE_PASS (9/9)',
    'dir': 'D:/models/upload/',
    'files': {
        'Track2_M24_v3.zip': {'gib': 1.67, 'role': '① 校准弹 (0.5B 基准)'},
        'Track2_M0S7_SC10_v3.zip': {'gib': 3.34, 'role': '② σ 测量'},
        'Track2_M0S1_SC093_v3.zip': {'gib': 3.37, 'role': '③ lpips 主轴判别 (核心)'},
        'Track2_G3W18S1_SC093_v3.zip': {'gib': 3.35, 'role': '④ dists 判别'},
        'Track2_M0S7_SC097_v3.zip': {'gib': 3.34, 'role': '波2 T高首选 (测试集直证第一)'},
    },
    'naming_rule': {
        'decided_by': '用户 2026-09-11',
        'rule': '文件名不要队名/电话后缀, 形如 Track2_M24_v3.zip',
        'note': '改名不影响 zip 内容, 也不需要重新过门禁(只要新名仍纯 ASCII)',
    },
    'superseded': 'v2 (无 _vendor 加固) 已删 — 对抗审计下会重演 Failed to load IQAEvaluator',
    'gate': 'preupload_gate.py 9 项: ① 文件名/内部路径 ASCII ② 语法兼容 ③ 结构/体积/出图尺寸 '
            '④ 接口 512 推理 ⑤ 标准 IQA 遮蔽 ⑥ 对抗 IQA 遮蔽',
}

P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('台账已更新: _upload_ready -> v3 无后缀 5 包')
for k, v in d['_upload_ready']['files'].items():
    print(f'  {v["gib"]:>5} GiB  {k}   {v["role"]}')
print('命名规则:', d['_upload_ready']['naming_rule']['rule'])
