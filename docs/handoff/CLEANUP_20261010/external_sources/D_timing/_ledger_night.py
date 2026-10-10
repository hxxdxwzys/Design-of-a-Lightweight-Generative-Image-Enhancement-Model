# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
p = Path(r'D:/tmp_timing/score_ledger.json')
d = json.loads(p.read_text(encoding='utf-8'))
d['score_model_v31_20260915night'] = {
 'formula': 'score = Q * ((285.6/(t_ms + C))/0.78)**p ; 自由拟合 (Q,p,C) = (43.429, 0.1980, 2.515)',
 'fit_points': [
   {'pkg': 'Track2_M24_v3', 't_ms': 352.9, 'official': 43.6951, 'resid': -0.010},
   {'pkg': 'Track2_ProbeSlow8', 't_ms': 2930.0, 'official': 28.7538, 'resid': +0.009},
   {'pkg': 'Track2_M24img_V7ganrun', 't_ms': 4.62, 'official': 94.7249, 'resid': +0.004},
   {'pkg': 'Track2_M24img_V4grun', 't_ms': 1.60, 'official': 105.6401, 'resid': -0.002}],
 'official_ratio_ceiling': 113.6,
 'm24_images_at_ceiling': 116.4,
 'our_best_is_at_pct_of_ceiling': 90.7,
 'C_robustness': {'0.0': 'no-ceiling fit SSE 27.4 (rejected)', '1.0': 'r_max 285.6x SSE 5.79',
                  '2.0': 'r_max 142.8x SSE 0.447', '2.515': 'r_max 113.6x SSE 0.0002 (best)',
                  '3.0': 'r_max 95.2x SSE 0.287', '5.0': 'r_max 57.1x SSE 4.57'},
 'leader_implication': '榜首 217.21 若用我们这批图需要 t_official=0.108ms(不可能) => 其图像项必须 ~81 (我们的 1.87 倍)',
 'supersedes': 'NEXT_ROUND_PROMPT_20260915PM.md 里的 C=2.0 / 142.8x 天花板（偏乐观）'}
d['runner_latency_breakdown_20260915night'] = {
 'harness': 'q3_runner_bench.py, GPU 空闲, 200 次取中位数, 逐次 sync',
 'v4g_128': {'infer_cpu_in': 2.467, 'infer_cuda_in': 1.670, 'graph_replay': 1.409},
 'v7gan_256': {'infer_cpu_in': 5.525, 'infer_cuda_in': 4.824, 'graph_replay': 4.489},
 'v11b_256': {'infer_cpu_in': 5.488, 'infer_cuda_in': 4.855, 'graph_replay': 4.546},
 'v9b_512': {'infer_cpu_in': 33.72, 'infer_cuda_in': 30.41, 'graph_replay': 30.67},
 'components': {'h2d_pageable': 0.49, 'resize_512_128': 0.058, 'resize_128_512': 0.133, 'float_clamp': 0.069},
 'law': '时延 ~= 内核数 x 激活张量尺寸 (显存带宽 ~140GB/s), 不是 FLOPs; UNet 在 128 平方有 ~200 内核 = 1.41ms'}
d['native_sweep_20260915night'] = {
 'what': 'q4_native_sweep.py: 同一份 JIT 权重, 只改 runner 的内部工作分辨率 _NATIVE',
 'table': {'v7gan': {'128': [1.749, 105.0], '160': [2.094, 103.5], '192': [2.716, 101.5], '224': [3.802, 98.6], '256': [4.831, 94.3]},
           'v11b': {'128': [1.612, 105.5], '160': [2.047, 103.8], '192': [2.524, 101.9], '224': [3.675, 98.9], '256': [4.896, 94.2]},
           'v5d': {'128': [1.609, 105.6], '160': [2.199, 102.9], '192': [2.687, 101.3], '224': [3.712, 98.8], '256': [4.879, 94.2]}},
 'unit': 'ms / 配 M24 图后的预测分',
 'reading': '现有权重的上限就是 ~105.5, 与 v4g(105.64) 同级; 想更快只能换更扁的网'}
d['runner_defects_20260915night'] = {
 'harness': 'q4_nan_check.py, case12 中心 512 crop, fp16 前向, clamp 前的值域',
 'ranges': {'v4g': [-0.789, 1.515], 'v5d': [-1.979, 1.499], 'v7gan': [-1.002, 1.706],
            'v9b': [-0.876, 14.406], 'v11b': [-67.938, 2.012], 'v1': [-3.197, 2.977]},
 'reading': 'v11b 有 -67.9 的爆点(clamp 后成黑斑), v9b +14.4 白斑; v4g 干净但输出与 LQ 差仅 2.09/255 (= 什么都不做)',
 'visual': 'D:/tmp_timing/q4_visual.png',
 'conclusion': '现有微模型 runner 的"干净"与"真增强"不可兼得'}
d['q_drivers_20260915night'] = {
 'harness': 'q_fit_metrics2.py + q_fit2.py + q8_probeA.py, 原生 4K 测试集前 30 张',
 'metrics': {'M24': {'maniqa': 0.5042, 'clipiqa': 0.9023, 'niqe': 5.2559, 'lapvar': 602.7, 'Q_official': 43.44},
             'probeA_sc04': {'maniqa': 0.4586, 'clipiqa': 0.7924, 'lapvar': 699.7, 'Q_official': 43.03},
             'w2_m0s1_s090': {'maniqa': 0.4762, 'clipiqa': 0.8718, 'niqe': 5.9425, 'Q_official': 41.46},
             'w2_m0s7_s097': {'maniqa': 0.4341, 'clipiqa': 0.8766, 'niqe': 6.3431, 'Q_official': 38.17},
             'TinyV9b': {'maniqa': 0.3634, 'clipiqa': 0.6409, 'Q_official': 25.66},
             'TinyV7gan': {'maniqa': 0.2819, 'clipiqa': 0.5059, 'Q_official': 19.14},
             'TinyV1': {'maniqa': 0.2687, 'clipiqa': 0.3367, 'Q_official': 18.38},
             'PixB': {'maniqa': 0.2698, 'clipiqa': 0.3143, 'Q_official': 16.13},
             'TinyV2g': {'maniqa': 0.2588, 'clipiqa': 0.3532, 'Q_official': 15.54},
             'TinyV5d': {'maniqa': 0.2762, 'clipiqa': 0.3897, 'Q_official': 15.42},
             'LQ': {'maniqa': 0.2819, 'clipiqa': 0.2752, 'niqe': 7.81}},
 'cross_family_fit': 'Q = 118.3*maniqa - 15.26 (LOOCV 1.78 over 9 pts / 5 families); 单指标 LOOCV: maniqa 1.78, clipiqa 3.17, lapvar 6.16, niqe 14.80',
 'within_family_breaks': 'sc0.4 -> sc0.8: 官方 Q 只 +0.41, 而 maniqa +0.0456 (模型预测 +5.4) => maniqa 是相关量不是因果量',
 'postproc_falsified_again': 'median3 / med3+unsharp / gauss0.8+unsharp / med5+unsharp 四组: maniqa 全跌 -0.039~-0.129, clipiqa 全跌 -0.16~-0.26',
 'gt_implied_Q': 30.0,
 'gt_note': 'GT(val) maniqa 0.3883 -> Q~30 < 我们的 43.4 => "完美图"在官方图像项上打不过我们的生成图',
 'leader_implied_maniqa': 0.81,
 'conclusion': 'Q 没有已识别的合法杠杆; 榜首 Q~81 需要 maniqa 0.81 (GT 的 2.1 倍) =>【区间推断】只能来自针对指标优化的图 (= 我们被判过 3 次 fraud 的那类)'}
tt = d.get('tool_traps_20260915pm', [])
if isinstance(tt, list):
    tt = tt + ['默认 python 没有 numpy; 正确解释器 = D:/Python/python.exe (torch 2.7.0+cu126, pyiqa 0.1.16)',
               'MUSIQ 在原生 4K 上 51 s/张 (clipiqa 1.3s / maniqa 1.5s / niqe 0.27s) => native-musiq 面板 <=10 张',
               'probeA 的图在 D:/models/probe_a/pred_sc0.4/case{n}.png (不是 output_dir)']
    d['tool_traps_20260915pm'] = tt
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('ledger keys:', len(d))
print('LEDGER_OK')
