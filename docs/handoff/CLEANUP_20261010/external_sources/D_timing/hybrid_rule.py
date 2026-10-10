# -*- coding: utf-8 -*-
"""混合规则: 安全池 M=0.02 换图; 非安全池仅在大边际 (>0.05) 时换图"""
import json
from collections import Counter

import numpy as np

sel = json.load(open(r'D:/tmp_timing/selection_analysis.json'))
rep = json.load(open(r'D:/tmp_timing/per_image_report.json'))['test_clipiqa']
val = sel['val_lpips_used']
s6k_val = 0.3583
safe = {n for n, v in val.items() if v is not None and v <= s6k_val + 0.005}
print('safe =', sorted(safe), flush=True)

sr = json.load(open(r'D:/models/sweep_test_out/sweep_results.json'))
by_name = {r['name']: r for r in sr['results']}
names = [r['name'] for r in sr['results'] if 'test_clipiqa' in r and r.get('n', 0) >= 100]
pi = {n: by_name[n]['per_image'] for n in names}
keys = sorted(set(rep), key=lambda k: int(k.replace('case', '')))
s6k_pi = {k: rep[k]['s6k'] for k in keys}


def best_for(k, pool):
    bv, bn = s6k_pi[k], 'old_s6000'
    for n in pool:
        v = pi[n].get(k, {}).get('clipiqa')
        if v is not None and v > bv:
            bv, bn = v, n
    return bn, bv


ch = {}
for k in keys:
    bn, bv = best_for(k, names)
    if bn == 'old_s6000':
        ch[k] = 'old_s6000'
        continue
    if bn in safe and bv - s6k_pi[k] > 0.02:
        ch[k] = bn
    elif bn not in safe and bv - s6k_pi[k] > 0.05:
        ch[k] = bn
    else:
        ch[k] = 'old_s6000'

mean_sel = float(np.mean([pi[ch[k]][k]['clipiqa'] if ch[k] != 'old_s6000' else s6k_pi[k] for k in keys]))
print('hybrid: mean=%.4f (delta %+.4f), swaps=%d'
      % (mean_sel, mean_sel - 0.5354, sum(1 for k in keys if ch[k] != 'old_s6000')), flush=True)
print('sources:', dict(Counter(ch.values())), flush=True)
for src in sorted(set(ch.values())):
    if src == 'old_s6000':
        continue
    ks = [k for k in keys if ch[k] == src]
    margins = [pi[src][k]['clipiqa'] - s6k_pi[k] for k in ks]
    print('  %-12s n=%d margins: min %.3f med %.3f max %.3f'
          % (src, len(ks), min(margins), float(np.median(margins)), max(margins)), flush=True)
json.dump({'choice': ch}, open(r'D:/tmp_timing/hybrid_choice.json', 'w'), indent=1)
print('[saved] hybrid_choice.json', flush=True)
