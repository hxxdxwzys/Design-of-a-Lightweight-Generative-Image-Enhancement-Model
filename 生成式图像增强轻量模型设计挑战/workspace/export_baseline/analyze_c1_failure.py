# -*- coding: utf-8 -*-
"""C1 失败分析（仅 CPU 读盘）：把“fp16 迁移发散”与“导出/重载等价”两件事分开。

输入：已落盘的 phase-1 源参考 NPY、C1 readings、已验收 fp32（R-B）NPY。
输出：<run>/evidence/c1_failure_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import cv2

RUN = Path(sys.argv[1]).resolve()
SRC = RUN / 'evidence' / 'src_outputs'
RB = Path('D:/models/pipeline_baseline/PIPE01RB_20260917_021500/attempt_01')
TOL = json.loads((Path(__file__).resolve().parent / 'export_config.json')
                 .read_text(encoding='utf-8'))['tolerances']


def lowpass(a, sigma=5.0):
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma)


readings = json.loads((RUN / 'evidence' / 'paired_readings_c1.json').read_text(encoding='utf-8'))
by_name = {c['name']: c for c in readings['cases']}

out = {
    'schema': 'EXPORT01-c1-failure-analysis/1',
    'verdict': 'FAIL',
    'summary': 'fp16 迁移在 4K 上与原已验收 fp32 基线严重不等价（均值差 0.25/0.40），'
               '而导出/重载本身在 512 上逐位一致（max=mean=0）。另一独立事实：同一实例、'
               '同一输入重复推理在 512 上已有 max=0.047 的差异，说明纯 fp16 链存在运行间'
               '不确定性，预登记的 1e-5 同精度判据在纯 fp16 下不可达。',
    'findings': [],
    'budget_state': {
        'top_level_infers_used': 22, 'top_level_infers_cap': TOL and 24,
        'n_4k_infers_used': 8, 'n_4k_infers_cap': 8,
        'note': '4K 额度已用尽，未再运行 4K；C2 未启动（C1 门槛未通过）',
    },
}

for name, ref_name in (('case1_full', 'attempt_01_072821_A_case1_full_r2.npy'),
                       ('case7_full', 'attempt_01_072821_A_case7_full_r3.npy')):
    a = np.load(SRC / ('%s_src.npy' % name), allow_pickle=False)[0].transpose(1, 2, 0)
    b = np.load(RB / ref_name, allow_pickle=False)[0].transpose(1, 2, 0)
    la, lb = lowpass(a), lowpass(b)
    ha, hb = a - la, b - lb
    c = by_name[name]
    out['findings'].append({
        'id': 'F1_fp16_vs_fp32',
        'case': name,
        'fp16_src_vs_fp32_max_abs': c['fp32_ref']['max_abs'],
        'fp16_src_vs_fp32_mean_abs': c['fp32_ref']['mean_abs'],
        'high_freq_energy': {'fp16': float(np.abs(ha).mean()), 'fp32': float(np.abs(hb).mean())},
        'high_freq_ratio': float(np.abs(ha).mean() / np.abs(hb).mean()),
        'diagnostic_thresholds': {
            'max_abs': TOL['fp16_vs_fp32_max_abs_diagnostic'],
            'mean_abs': TOL['fp16_vs_fp32_mean_abs_diagnostic']},
        'passed': False,
    })
    out['findings'].append({
        'id': 'F2_export_reload_equivalence',
        'case': name,
        'eager_fp16_vs_reloaded_max_abs': c['same_precision_max_abs'],
        'eager_fp16_vs_reloaded_mean_abs': c['same_precision_mean_abs'],
        'criterion_max_abs': TOL['same_precision_max_abs'],
        'passed': False,
    })

c512 = by_name['case1_crop512']
out['findings'].append({
    'id': 'F2_export_reload_equivalence', 'case': 'case1_crop512',
    'eager_fp16_vs_reloaded_max_abs': c512['same_precision_max_abs'],
    'eager_fp16_vs_reloaded_mean_abs': c512['same_precision_mean_abs'],
    'criterion_max_abs': TOL['same_precision_max_abs'], 'passed': True,
})
out['findings'].append({
    'id': 'F3_run_to_run_nondeterminism',
    'case': 'case1_crop512',
    'same_instance_same_input_repeat_max_abs': c512.get('repeat_max_abs'),
    'repeat_png_identical': c512.get('repeat_png_identical'),
    'implication': '纯 fp16（权重+激活 fp16，无 autocast）在 RTX4060/cuDNN 下运行间不可复现；'
                   'fp32 基线在同一环境下逐位可复现（R-B 已验收 max=mean=0）。'
                   '预登记的 1e-5 同精度判据在纯 fp16 下结构性不可达，需要 Architect 重新裁定'
                   '精度配方与判据，而不是就地放宽阈值。',
})
out['not_established'] = [
    '具体发散子模块（VAE 编码 1024 tile / 解码 128 tile / DiT）——需要新的 4K 预算做分段探针',
    'autocast（混合精度，softmax/norm 用 fp32）是否能同时满足 fp16 交付要求与可复现性',
    '官方测速对“fp16”的具体判定口径（仅权重 fp16 还是允许 autocast）',
]
(RUN / 'evidence' / 'c1_failure_analysis.json').write_text(
    json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps({'verdict': out['verdict'], 'n_findings': len(out['findings']),
                  'budget': out['budget_state']}, ensure_ascii=False))
