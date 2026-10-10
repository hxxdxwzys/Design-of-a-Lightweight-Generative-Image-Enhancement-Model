# -*- coding: utf-8 -*-
"""sweep_test_all 结果分析 (2026-08-14)

输入:
  D:/models/sweep_test_out/sweep_results.json  (逐 ckpt 测试集 clipiqa/niqe, per-image)
  D:/tmp_timing/per_image_report.json          (LQ/s6k/HYPIR 逐图 clipiqa 参考)
  D:/tmp_timing/sweep_f2k.json                 (f2k val lpips/clipiqa)
  D:/tmp_timing/cnn_val_s*/val_result.json     (旧 run val 结果, 部分)
输出:
  D:/tmp_timing/sweep_analysis.json + 控制台汇总
"""
import json
from pathlib import Path

import numpy as np

SR = Path(r'D:/models/sweep_test_out/sweep_results.json')
REP = Path(r'D:/tmp_timing/per_image_report.json')
F2K_VAL = Path(r'D:/tmp_timing/sweep_f2k.json')

OLD_VAL_FILES = {
    'old_s3000': 'cnn_val_s3k', 'old_s5000': 'cnn_val_s5k', 'old_s7000': 'cnn_val_s7000',
    'old_s8000': 'cnn_val_s8000', 'old_s9000': 'cnn_val_s9k', 'old_s10000': 'cnn_val_s10k',
}
REF_NAME = 'old_s6000'


def main():
    res = json.load(open(SR))['results']
    by_name = {r['name']: r for r in res}
    rep_tc = json.load(open(REP))['test_clipiqa']
    if REF_NAME in by_name and 'test_clipiqa' in by_name[REF_NAME]:
        ref = by_name[REF_NAME]
        ref_cq = ref['test_clipiqa']
        ref_pi = ref['per_image']
        print(f'[ref] 用 sweep 内 old_s6000: test_clipiqa={ref_cq:.4f}', flush=True)
    else:
        # 后备: per_image_report 的 s6k 逐图值 (同一 ckpt/输出/度量口径)
        ref_pi = {k: {'clipiqa': v['s6k']} for k, v in rep_tc.items()}
        ref_cq = float(np.mean([x['clipiqa'] for x in ref_pi.values()]))
        ref = {'test_clipiqa': ref_cq, 'test_niqe': None, 'per_image': ref_pi}
        print(f'[ref] 用 per_image_report s6k: test_clipiqa={ref_cq:.4f}', flush=True)

    lq_pi = {k: v['lq'] for k, v in rep_tc.items()}

    f2k_val = {}
    if F2K_VAL.exists():
        for r in json.load(open(F2K_VAL)):
            f2k_val['f2k_' + r['ckpt'].replace('.pt', '')] = {'val_lpips': r['lpips'], 'val_clipiqa': r['clipiqa']}
    for name, d in OLD_VAL_FILES.items():
        p = Path(rf'D:/tmp_timing/{d}/val_result.json')
        if p.exists():
            v = json.load(open(p))
            f2k_val.setdefault(name, {})['val_lpips'] = v['lpips']
            f2k_val.setdefault(name, {})['val_clipiqa'] = v['clipiqa']

    rows = []
    for name, r in sorted(by_name.items(), key=lambda kv: -kv[1].get('test_clipiqa', -1)):
        if 'test_clipiqa' not in r or r['n'] < 100:
            continue
        pi = r['per_image']
        keys = sorted(set(pi) & set(ref_pi), key=lambda k: int(k.replace('case', '')))
        diffs = [pi[k]['clipiqa'] - ref_pi[k]['clipiqa'] for k in keys]
        d = np.array(diffs)
        better_frac = float((d > 0).mean())
        worse_frac = float((d < 0).mean())
        v = f2k_val.get(name, {})
        rows.append({
            'name': name,
            'test_clipiqa': r['test_clipiqa'],
            'test_niqe': r.get('test_niqe'),
            'delta_vs_s6k': round(r['test_clipiqa'] - ref_cq, 4),
            'mean_pair_diff': round(float(d.mean()), 4),
            'better_frac': round(better_frac, 3),
            'worse_frac': round(worse_frac, 3),
            'val_lpips': v.get('val_lpips'),
            'val_clipiqa': v.get('val_clipiqa'),
        })

    print('===== 全 ckpt 测试集对比 (按 test clipiqa 降序; s6k=old_s6000) =====')
    print(f"  参考 s6k: test_clipiqa={ref_cq:.4f} niqe={ref.get('test_niqe')}")
    for r in rows:
        extra = f" val_lpips={r['val_lpips']:.4f}" if r['val_lpips'] is not None else " val_lpips=?"
        print(f"  {r['name']:14s} cq={r['test_clipiqa']:.4f} (Δ{r['delta_vs_s6k']:+.4f}) "
              f"niqe={r['test_niqe']:.2f} better={r['better_frac']:.2f} worse={r['worse_frac']:.2f}{extra}")

    # 候选: 测试 clipiqa 高于 s6k 且 val lpips 不显著退步 (<= +0.01)
    print('\n===== 候选 (test_clipiqa > s6k + 0.005 且 val_lpips <= s6k_val + 0.01) =====')
    s6k_val_lp = f2k_val.get('old_s6000', {}).get('val_lpips', 0.358)
    cands = [r for r in rows if r['delta_vs_s6k'] > 0.005 and (r['val_lpips'] is None or r['val_lpips'] <= s6k_val_lp + 0.01)]
    for r in cands:
        print(f"  ✓ {r['name']}: Δcq={r['delta_vs_s6k']:+.4f} niqe={r['test_niqe']:.2f} val_lpips={r['val_lpips']}")

    # 逐图: 每个 ckpt 在多少图上超过 s6k (及超过 LQ)
    print('\n===== 逐图胜出统计 (vs s6k) =====')
    for r in rows[:12]:
        print(f"  {r['name']:14s} 赢 {int(r['better_frac']*100):3d}/100 图, 输 {int(r['worse_frac']*100):3d}/100 图")

    # 逐图最优 ckpt (仅看 clipiqa, 评估 per-image 选择的天花板)
    names = [r['name'] for r in rows]
    pi_maps = {n: by_name[n]['per_image'] for n in names}
    keys = sorted(set(pi_maps[REF_NAME]) & set(rep_tc), key=lambda k: int(k.replace('case', '')))
    per_best = {}
    per_best_delta = {}
    for k in keys:
        best, bestv = REF_NAME, ref_pi[k]['clipiqa']
        for n in names:
            v = pi_maps[n].get(k, {}).get('clipiqa')
            if v is not None and v > bestv:
                best, bestv = n, v
        per_best[k] = best
        per_best_delta[k] = bestv - ref_pi[k]['clipiqa']
    sel_mean = float(np.mean(list(per_best_delta.values())))
    print(f'\n逐图最优选择 (top-1/图, clipiqa 上限): 均值比 s6k 高 {sel_mean:+.4f}')
    from collections import Counter
    cnt = Counter(per_best.values())
    print('逐图最优 ckpt 分布:', dict(cnt.most_common(8)))

    out = {'ref_s6k': {'test_clipiqa': ref_cq, 'test_niqe': ref.get('test_niqe')},
           'rows': rows, 'candidates': cands, 'per_best_count': dict(cnt),
           'per_best_mean_delta': sel_mean}
    with open(r'D:/tmp_timing/sweep_analysis.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\n[saved] D:/tmp_timing/sweep_analysis.json')


if __name__ == '__main__':
    main()
