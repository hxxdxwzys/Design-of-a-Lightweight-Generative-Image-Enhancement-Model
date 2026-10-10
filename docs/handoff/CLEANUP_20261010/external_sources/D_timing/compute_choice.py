# -*- coding: utf-8 -*-
"""导出全池逐图选择映射 (各边际) -> choice_maps.json
之后 v2b sweep 完成后重跑本脚本 + hybrid_rule.py 即可更新混合策略。
"""
import json

import numpy as np

rep = json.load(open(r'D:/tmp_timing/per_image_report.json'))['test_clipiqa']
sr = json.load(open(r'D:/models/sweep_test_out/sweep_results.json'))
by_name = {r['name']: r for r in sr['results']}
names = [r['name'] for r in sr['results'] if 'test_clipiqa' in r and r.get('n', 0) >= 100]
pi = {n: by_name[n]['per_image'] for n in names}
keys = sorted(set(rep), key=lambda k: int(k.replace('case', '')))
s6k_pi = {k: rep[k]['s6k'] for k in keys}

per_best_name, per_best_val = {}, {}
for k in keys:
    bv, bn = s6k_pi[k], 'old_s6000'
    for n in names:
        v = pi[n].get(k, {}).get('clipiqa')
        if v is not None and v > bv:
            bv, bn = v, n
    per_best_name[k], per_best_val[k] = bn, bv

maps = {}
for M in [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
    ch = {k: (per_best_name[k] if per_best_val[k] - s6k_pi[k] > M else 'old_s6000') for k in keys}
    mean = float(np.mean([per_best_val[k] if ch[k] != 'old_s6000' else s6k_pi[k] for k in keys]))
    n_swap = sum(1 for k in keys if ch[k] != 'old_s6000')
    maps['M%.3f' % M] = {'choice': ch, 'mean': round(mean, 4),
                         'delta': round(mean - 0.53538294672966, 4), 'n_swap': n_swap}
    print('M=%.3f: mean=%.4f delta=%+.4f swaps=%d' % (M, mean, mean - 0.53538294672966, n_swap), flush=True)

json.dump({'maps': maps, 's6k_mean': 0.53538294672966}, open(r'D:/tmp_timing/choice_maps.json', 'w'), indent=1)
print('[saved] choice_maps.json', flush=True)
