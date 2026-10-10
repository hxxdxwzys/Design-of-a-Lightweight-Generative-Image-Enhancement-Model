# -*- coding: utf-8 -*-
"""HYPIR vs 当前 M10 混合 逐图 clipiqa 对照 (CPU, 只读分析):
找出 HYPIR 输出比当前混合逐图更优的图 (可作为池外候选, 待用户决定是否并入)。
"""
import json
from collections import Counter

rep = json.load(open(r'D:/tmp_timing/per_image_report.json'))['test_clipiqa']
cm = json.load(open(r'D:/tmp_timing/choice_maps.json'))
ch = cm['maps']['M0.010']['choice']
sr = json.load(open(r'D:/models/sweep_test_out/sweep_results.json'))
pi = {r['name']: r['per_image'] for r in sr['results'] if 'per_image' in r}

keys = sorted(ch, key=lambda k: int(k.replace('case', '')))
wins, wins_m5 = [], []
rows = []
for k in keys:
    src = ch[k]
    mix_v = pi[src][k]['clipiqa'] if src != 'old_s6000' else rep[k]['s6k']
    hyp_v = rep[k]['hypir']
    rows.append((k, src, mix_v, hyp_v, hyp_v - mix_v))
    if hyp_v > mix_v:
        wins.append(k)
        if hyp_v - mix_v > 0.005:
            wins_m5.append(k)

import numpy as np
print('HYPIR 赢当前混合:', len(wins), '张; 边际>0.005:', len(wins_m5), '张', flush=True)
print('若把 HYPIR 赢图全换 (clip 视角): mean %+.4f' % float(np.mean([hyp_v for _, _, _, hyp_v, _ in rows])))
mix_mean = float(np.mean([v for _, _, v, _, _ in rows]))
hyp_sel_mean = float(np.mean([hyp_v if hyp_v > mix_v else mix_v for _, _, mix_v, hyp_v, _ in rows]))
print(f'mix mean={mix_mean:.4f} -> hyp 择优 mean={hyp_sel_mean:.4f} (d={hyp_sel_mean - mix_mean:+.4f})', flush=True)
for k, src, mv, hv, d in sorted(rows, key=lambda r: -r[4])[:12]:
    print(f'  {k}: mix({src})={mv:.4f} hyp={hv:.4f} d={d:+.4f}', flush=True)
json.dump({'hypir_wins': wins, 'hypir_wins_m5': wins_m5},
          open(r'D:/tmp_timing/hypir_vs_mix.json', 'w'), indent=1)
print('[saved] hypir_vs_mix.json', flush=True)
