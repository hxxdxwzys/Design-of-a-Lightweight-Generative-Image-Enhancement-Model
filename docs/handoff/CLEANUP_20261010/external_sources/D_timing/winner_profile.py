# -*- coding: utf-8 -*-
"""逐图最优 ckpt 的退化画像交叉分析: f2k_s1000 等赢家赢在哪类图上 (评估混合的 lpips 风险)"""
import json
from collections import defaultdict

import numpy as np

sel = json.load(open(r'D:/tmp_timing/selection_analysis.json'))
deg = json.load(open(r'D:/tmp_timing/degradation_analysis.json'))
blk = json.load(open(r'D:/tmp_timing/jpeg_blockiness.json'))
rep = json.load(open(r'D:/tmp_timing/per_image_report.json'))['test_clipiqa']

byn = {r['name']: r for r in deg['per_image'] if r['tag'] == 'test'}
bynb = {r['name']: r for r in blk['per_image'] if r['tag'] == 'test'}
pb = sel['ceiling_all']['per_image']

groups = defaultdict(list)
for k, ck in pb.items():
    d = byn.get(k + '.jpg', {})
    b = bynb.get(k + '.jpg', {})
    groups[ck].append({'k': k, 'sigma': d.get('est_sigma'), 'noise': d.get('est_noise_n'),
                       'jpeg_q': b.get('est_jpeg_q'), 'lq_cq': rep[k]['lq'], 's6k_cq': rep[k]['s6k']})

print('逐图最优 ckpt 分布 (全池天花板):')
for ck, imgs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    sig = np.array([x['sigma'] for x in imgs if x['sigma'] is not None])
    noi = np.array([x['noise'] for x in imgs if x['noise'] is not None])
    lqc = np.array([x['lq_cq'] for x in imgs])
    print('  %-14s n=%3d  sigma_med=%.2f  noise_med=%.2f  LQcq_med=%.3f'
          % (ck, len(imgs), float(np.median(sig)), float(np.median(noi)), float(np.median(lqc))))

print()
for ck in ['f2k_s1000', 'old_s9000', 'f2k_s7000', 'old_s8000', 'f2k_s9000', 'old_s5000', 'f2k_s2000']:
    if ck not in groups:
        continue
    imgs = sorted(groups[ck], key=lambda x: -x['s6k_cq'])
    print('--- %s 赢的 %d 张 (按 s6k_clipiqa 降序, 前 12) ---' % (ck, len(imgs)))
    for x in imgs[:12]:
        print('  %s: sigma=%s noise=%s jpeg_q=%s LQcq=%.3f s6kcq=%.3f'
              % (x['k'], x['sigma'], x['noise'], x['jpeg_q'], x['lq_cq'], x['s6k_cq']))

# 汇总: 非 s6k 赢家图的画像 vs s6k 赢家图
other = [x for ck, imgs in groups.items() if ck != 'old_s6000' for x in imgs]
s6k_g = groups.get('old_s6000', [])
def stats(lst):
    sig = np.array([x['sigma'] for x in lst if x['sigma'] is not None])
    noi = np.array([x['noise'] for x in lst if x['noise'] is not None])
    lqc = np.array([x['lq_cq'] for x in lst])
    return len(lst), float(np.median(sig)), float(np.median(noi)), float(np.median(lqc))
print()
print('被替换图 (非 s6k 赢) n=%d sigma_med=%.2f noise_med=%.2f LQcq_med=%.3f' % stats(other))
print('s6k 赢图            n=%d sigma_med=%.2f noise_med=%.2f LQcq_med=%.3f' % stats(s6k_g))
