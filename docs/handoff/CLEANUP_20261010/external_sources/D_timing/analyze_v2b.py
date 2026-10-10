# -*- coding: utf-8 -*-
"""v2b sweep 结果分析 (只读, sweep 完成后运行):
对每个 v2b ckpt: test clipiqa 均值 / Δ vs s6k / 逐图胜场数;
并给出全池 (含 v2b) 在各边际 M 下的选择映射期望。
用法: D:/Python/python.exe analyze_v2b.py
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
S6K_MEAN = 0.53538294672966

v2b = [n for n in names if n.startswith('metric_net_v2b')]
print('=== v2b ckpt vs s6k ===')
for n in sorted(v2b):
    m = by_name[n]['test_clipiqa']
    wins = [k for k in keys if pi[n].get(k, {}).get('clipiqa', 0) > s6k_pi[k]]
    w5 = [k for k in wins if pi[n][k]['clipiqa'] - s6k_pi[k] > 0.05]
    print(f'{n:28s} clipiqa={m:.4f} (d={m - S6K_MEAN:+.4f})  wins={len(wins):3d}  wins>0.05={len(w5):3d}')

print('\n=== 全池逐图最优 (含 v2b) 边际表 ===')
per_best_name, per_best_val = {}, {}
for k in keys:
    bv, bn = s6k_pi[k], 'old_s6000'
    for n in names:
        v = pi[n].get(k, {}).get('clipiqa')
        if v is not None and v > bv:
            bv, bn = v, n
    per_best_name[k], per_best_val[k] = bn, bv

for M in [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
    ch = {k: (per_best_name[k] if per_best_val[k] - s6k_pi[k] > M else 'old_s6000') for k in keys}
    mean = float(np.mean([per_best_val[k] if ch[k] != 'old_s6000' else s6k_pi[k] for k in keys]))
    n_swap = sum(1 for k in keys if ch[k] != 'old_s6000')
    v2b_swap = sum(1 for k in keys if ch[k] != 'old_s6000' and ch[k].startswith('metric_net_v2b'))
    print(f'M={M:.3f}: mean={mean:.4f} delta={mean - S6K_MEAN:+.4f} swaps={n_swap:3d} (v2b={v2b_swap:2d})')

# v2b 增量归因: 去掉 v2b 后的边际表 (与旧 M01 基线比)
print('\n=== 无 v2b 池边际表 (对照旧 M01) ===')
pool_old = [n for n in names if not n.startswith('metric_net_v2b')]
for M in [0.005, 0.01, 0.015]:
    ch = {}
    for k in keys:
        bv, bn = s6k_pi[k], 'old_s6000'
        for n in pool_old:
            v = pi[n].get(k, {}).get('clipiqa')
            if v is not None and v > bv:
                bv, bn = v, n
        ch[k] = bn if bv - s6k_pi[k] > M else 'old_s6000'
    mean = float(np.mean([pi[ch[k]][k]['clipiqa'] if ch[k] != 'old_s6000' else s6k_pi[k] for k in keys]))
    print(f'M={M:.3f}: mean={mean:.4f} delta={mean - S6K_MEAN:+.4f} swaps={sum(1 for k in keys if ch[k] != "old_s6000"):3d}')
print('[done]', flush=True)
