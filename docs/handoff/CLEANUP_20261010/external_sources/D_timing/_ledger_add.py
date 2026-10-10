# -*- coding: utf-8 -*-
import json, shutil, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
p = Path('D:/tmp_timing/score_ledger.json')
shutil.copy2(p, p.with_suffix('.json.bak'))
d = json.loads(p.read_text(encoding='utf-8'))
print('top-level keys:', list(d.keys())[:40] if isinstance(d, dict) else type(d))
blk = {
  'when': '2026-09-15 上午',
  'kind': 'decision',
  'question': '评估在跑训练的价值并做决策',
  'valuations': {
    'v11b': {'verdict': 'keep_running', 'eta': '~11:05 (25k 步 @0.40s)',
             'pred': '256域推理+512域损失+band-limitD; 观感>=0.58 概率~35% -> ~47-48; 观感>=0.64 概率15-20% -> ~54'},
    'v9b2': {'verdict': 'cancel_defer', 'cost_h': 8.4, 'pred_gain': '+0.5~1.5',
             'why': 'L 在 0.64 以上只剩 23 分/单位; v7gan 轨迹本身在减速'},
    'chain_J': {'verdict': 'cancel_defer', 'why': '其章程自述不得复活保真轴, 产出不能改变任何允许的决策'},
    'H3_diag': {'verdict': 'promote_and_expand', 'why': 'MUSIQ 原生4K 109s/张 => 原设计要 5.5h; 且口径未扫过'}
  },
  'new_measurements': {
    'iqa_resolution_sensitivity_M24case1': {
      'clipiqa': {'native4K': 0.8791, 'ms1024': 0.8368, 'ms512': 0.4266, 'sec': [3.1, 0.1, 0.1]},
      'musiq':   {'native4K': 42.7314, 'ms1024': 57.4155, 'ms512': 61.3215, 'sec': [108.9, 0.2, 0.1]},
      'maniqa':  {'native4K': 0.4536, 'ms1024': 0.4823, 'ms512': 0.4410, 'sec': [3.6, 3.8, 3.7]},
      'niqe':    {'native4K': 5.8494, 'ms1024': 5.3267, 'ms512': 5.7714, 'sec': [1.5, 0.1, 0.1]},
      'implication': '官方 IQAEvaluator 不可能在原生 4K 上算 MUSIQ(100张=3h) => 官方大概率 resize; 我们此前代理口径可能选错'
    },
    'vram_under_concurrent_probe': '训练中(3.2GB)加载 pyiqa 并算一张 4K 图 => 7853/8188 MiB, 步时 0.34->0.40 s/step',
    'image_term_decomposition': {
      'model': 'score = IT(图像项) + S(ratio), S(0.78x)=0, S(r>=0.78x)=2.63*log2(r/0.78)',
      'IT_M24_images': 43.6951,
      'cross_check_same_images': {'M24@0.78x': 43.70, 'avg2@0.382x': 41.79, 'ProbeSlow8@0.098x': 43.72},
      'IT_vs_native_clipiqa_pearson_r': 0.973,
      'n_points': 11,
      'note': '与台账旧结论"官方质量项!=任何本地指标"相反: 那是没扣速度项造成的; 扣掉后 clipiqa 解释力很强'
    },
    'unexplained': '两个账号B 旧读数(probe_fast2 48.06 / Speed204x 51.89)比模型高 12~14; 48分以上仍未验证'
  },
  'actions': ['chain_20260915k.py 已启动: 等 v11b 收尾 -> 掐 v9b2 -> 测 1.4B 家族 512 时延 -> 跑 diag_panel v2',
              'diag_panel.py v2: 21 个图像集 x 4 口径(native/ms1024/ms512/c1024) x 4 指标 + lapvar, 增量 JSON + 自动相关分析',
              'train_tiny_iqa.py: 可微 IQA 感知损失变体(--w_iqa/--iqa_metric/--iqa_res), 仅当面板证明某指标是官方轴时才启用',
              'image_term.py: 官方分的图像项/速度项分解(用同图实测点标定 S)',
              '待定_解耦探测/ 增加 V7ganrun 版(预测 60.3); README 写明"最多只打一发"'],
  'submissions_pending_user': ['upload/Track2_TinyV9b.zip (合规, 预测 47-48)',
                               '待定: M24img+V7ganrun (预测 60.3) 或 M24img+V9brun (54.2), 只打一发']
}
d['decision_20260915b'] = blk
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('written keys:', len(d))
