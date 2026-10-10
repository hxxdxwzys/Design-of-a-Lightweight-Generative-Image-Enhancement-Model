# -*- coding: utf-8 -*-
"""清理后更新 asset_profiles.json 的路径引用 (2026-09-11)"""
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
    '_doc': '2026-09-11 清理后: 唯一上传位置 = D:/models/upload/ (5 个包, 全部 8/8 门禁 PASS); '
            'D:/models 顶层的同名副本已删(硬链接, 零字节损失); 提交历史 4 个失败原件已删(runner 快照已归档)',
    'dir': 'D:/models/upload/',
    'files': {
        'Track2_M24_v2_Yuki_18851225671.zip': {'gib': 1.67, 'role': '① 校准弹 (0.5B 基准)'},
        'Track2_M0S7_SC10_v2_Yuki_18851225671.zip': {'gib': 3.34, 'role': '② σ 测量'},
        'Track2_M0S1_SC093_v2_Yuki_18851225671.zip': {'gib': 3.37, 'role': '③ lpips 主轴判别 (核心)'},
        'Track2_G3W18S1_SC093_v2_Yuki_18851225671.zip': {'gib': 3.35, 'role': '④ dists 判别'},
        'Track2_M0S7_SC097_Yuki_18851225671.zip': {'gib': 3.34, 'role': '波2 T高首选 (测试集直证第一)'},
    },
    'naming_note': '队名/电话取自历史提交的 Yuki / 18851225671; 与报名信息不符请改名(改名不动 zip 内容)',
    'not_upload': {
        'C:/生成式图像增强轻量模型设计挑战/提交历史/*（已作废）.zip': '老世界 MetricMix/Mix 线归档 3 个 (1.53 GiB), 仅存档勿上传',
    },
}

d['_cleanup_20260911'] = {
    'freed_gib_est': 25.3,
    'deleted': [
        'D:/models/Track2_*.zip x5 (与 upload/ 硬链接, 释放 0 字节, 纯降噪)',
        '提交历史/Track2_*——.zip x4 (失败四发原件, 11.73 GiB; runner.py 已归档为 _失败原件_runner快照/runner_*_v1_FAILED.py)',
        'D:/models/wave2_*/render_png x4 (无损母版 6.76 GiB, 提交用 jpg 仍在 output_dir)',
        'D:/models/_harness_zip_sim (门禁临时解压 3.60 GiB, 自动重建)',
        'D:/models/wave2_m0s7_s097/model_dir (your_model.pt 裸副本 3.25 GiB, pack_model_dir 内有同一份)',
    ],
    'kept_on_purpose': [
        'D:/models/upload/ 5 个待上传包',
        'D:/models/wave2_*/output_dir (重打包所需)',
        'D:/models/final_compliant* (打包源: output_dir + model_dir)',
        '提交历史 3 个老世界归档 (1.53 GiB, 历史凭据)',
    ],
}

P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('已更新 asset_profiles.json: _upload_ready + _cleanup_20260911')
