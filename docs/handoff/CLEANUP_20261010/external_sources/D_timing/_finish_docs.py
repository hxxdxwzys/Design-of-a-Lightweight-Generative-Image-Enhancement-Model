# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
NL = chr(10)
R = Path(r'C:/生成式图像增强轻量模型设计挑战')
# 1) Q_CAMPAIGN 追加 §7
p = R / 'Q_CAMPAIGN_20260915.md'
t = p.read_text(encoding='utf-8')
s = Path(r'D:/tmp_timing/_campaign7.md').read_text(encoding='utf-8').replace('%%BT%%', chr(96))
if '## 7. Q2 热点定位' not in t:
    p.write_text(t.rstrip() + NL + s, encoding='utf-8'); print('campaign: appended 7')
else: print('campaign: 7 exists')
# 2) ledger 追加
lp = Path(r'D:/tmp_timing/score_ledger.json')
d = json.loads(lp.read_text(encoding='utf-8'))
d['q2_profile_20260915pm'] = {
  'tool': 'D:/tmp_timing/q2_profile.py',
  'op_cuda_time_6_infers': {'conv2d_s': 1.87, 'group_norm_s': 0.60, 'linear_addmm_bmm_s': 0.80, 'to_copy_s': 0.54},
  'copy_calls': 5124,
  'conclusion': 'VAE(conv+groupnorm) 比 0.5B DiT 更重; 854 次/次 dtype 搬运; 合法提速手段 = CUDA Graph + 统一 dtype',
  'prediction': '352.9ms -> 200~250ms => r 0.81->1.2~1.4 => 合规分 43.7 -> 48~52 (score_pred2.py)'
}
d['seam_finding_20260915pm'] = {
  'test_render_vertical_seam': {'base_sc0.8_t512': 1.054, 'sc1.2': 1.055, 'adain': 1.049},
  'val_render_vertical_seam_(old_world)': {'s108': 1.043, 's208': 0.977, 's308': 0.982, 'avg4': 0.957, 'pred_sc0.8': 0.998},
  'conclusion': '换尺度/换对齐都修不掉接缝; val 渲染(不同 VAE 分块)基本无接缝 => 真凶可能是 VAE 分块(推翻台账旧判断 latent 随机噪声)'}
d['tool_traps_20260915pm'] = ['vosr_lora_infer.py 的 torch.hub.load(dinov2) 要联网做 ref 解析 => 断网 60s 崩', 'measure_dir.py 只找 caseN.jpg => PNG 渲染必须 --pattern', 'measure_latency v2 偏悲观, 官方口径更接近 v1 的 200 次均值']
lp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('ledger keys=%d' % len(d))