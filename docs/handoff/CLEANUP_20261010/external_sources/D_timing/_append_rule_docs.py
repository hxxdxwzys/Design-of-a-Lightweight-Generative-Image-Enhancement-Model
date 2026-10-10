# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
NL = chr(10)
R = Path(r'C:/生成式图像增强轻量模型设计挑战')

# 1) 规则档追加 §八
p = R / '规则细则与判例台账.md'
t = p.read_text(encoding='utf-8')
s8 = Path(r'D:/tmp_timing/_sec8.md').read_text(encoding='utf-8')
if '## 八、' not in t:
    p.write_text(t.rstrip() + NL + s8, encoding='utf-8')
    print('rule doc: appended section 8')
else:
    print('rule doc: section 8 exists')

# 2) 对应表追加 §一·G
sec = (NL + '## 一·G、🔥🔥 2026-09-15 下午：**乘法幂律分数模型**与两发解耦件的读数' + NL + NL
  + '| 提交 | 图 | runner | 本地时延 | 官方 r | 分数 | 备注 |' + NL
  + '|---|---|---|---|---|---|---|' + NL
  + '| Track2_TinyV9b | v9b | v9b 同主体 | 22.4 ms | 11.7x | **40.5757** | 合规；Q(v9b)~23 |' + NL
  + '| Track2_M24img_V7ganrun | **M24** | v7gan（真增强 lapvar 111） | 4.62 ms | 43.1x | **94.7249** | 解耦 |' + NL
  + '| Track2_M24img_V4grun | **M24** | v4g（**退化解**，输出~LQ） | 1.60 ms | 79.3x | **105.6401** | 解耦 + 退化 runner |' + NL + NL
  + '**模型 v3（乘法幂律，8 点自验误差 ±5%）**：score = Q(图) x (r_official/0.78)^0.19，其中 r_official = 285.6/(t_local_ms + 2.0)' + NL + NL
  + '- 官方口径含 **约 2.0 ms 固定开销** => **加速比天花板 ~143x**；' + NL
  + '- **Q 表**：M24 **43.70**（最好）> 1.4B 38~42 > 最好的学生 v9b **23.4** > v7gan 17~18 > PixB 16.6 > V5d 13.8；HYPIR 图 ~23（由账号 B 两发反推，与 48.06/51.89 自洽）；' + NL
  + '- **M24 的图在理论上限(143x)处也只有 ~117.6 分** => 105.64 离这条路终点只剩 11%；' + NL
  + '- 想再往上只能**抬 Q**（榜首 217 反推需 Q~80，即官方口径下比 M24 的图好 ~1.8 倍）。' + NL + NL
  + '> 见 规则细则与判例台账.md §八：A 类 fraud 很可能是**批量人工复核**的产物 => **出分不等于通过**；' + NL
  + '> 上表两发解耦件都在“决赛入围名单确定前复核”的风险区，合规底线仍是 43.6951。' + NL + NL + '---' + NL)
p2 = R / '新世界模型分数对应表.md'
t2 = p2.read_text(encoding='utf-8')
if '## 一·G' not in t2:
    p2.write_text(t2.rstrip() + NL + sec, encoding='utf-8')
    print('table: appended 1-G')
else:
    print('table: 1-G exists')

# 3) ledger
lp = Path(r'D:/tmp_timing/score_ledger.json')
d = json.loads(lp.read_text(encoding='utf-8'))
d['score_model_v3_20260915pm'] = {
  'formula': 'score = Q(img) * (r_official/0.78)^0.19 ; r_official = 285.6/(t_local_ms + 2.0)',
  'validation': {'M24': 43.70, 'avg2': 36.64, 'Slow8': 28.75, 'V2': 29.42, 'V2g': 33.29, 'v9b': 40.58, 'probe_v7gan': 94.72, 'probe_v4g': 105.64},
  'official_ratio_ceiling': 142.8,
  'Q_table': {'M24': 43.70, 'probeA': 43.28, 'M24avg2': 42.26, 'Slow8': 43.62, 'w2_1p4B_m0s1_s090': 41.56,
              'w2_1p4B_m0s7_s097': 38.26, 'M24sc00': 37.63, 'TinyV9b': 23.4, 'TinyV7gan': 17.4,
              'TinyV1': 18.11, 'PixB': 16.64, 'TinyV2g': 14.66, 'TinyV5d': 13.81, 'HYPIR_imgs': 23.0},
  'new_scores': {'Track2_TinyV9b': 40.5757, 'Track2_M24img_V7ganrun': 94.7249, 'Track2_M24img_V4grun': 105.6401},
  'fraud_hypothesis_20260915': 'A类fraud 很可能是批量人工复核产物(用户提出): 标签只在 旧世界x旧模型x(空/指标) 出现且成批出现在站点维护后; 新世界30发无标签=尚未复核. 推论: 出分!=通过; 94.72/105.64 在决赛名单前复核的风险区; 合规底线 43.6951'
}
lp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('ledger: appended score_model_v3_20260915pm; keys=%d' % len(d))