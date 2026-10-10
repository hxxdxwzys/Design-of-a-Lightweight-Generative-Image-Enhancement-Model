# -*- coding: utf-8 -*-
"""逐图混合选择分析 (2026-08-14)
sweep 完成后运行: 基于测试集逐图 clipiqa, 评估
  1) 单 ckpt 最优
  2) 逐图选最优 ckpt 的天花板 (clipiqa 信号)
  3) 带边际的安全混合策略 (仅在 clipiqa 增益 > M 且所选 ckpt val lpips 安全时换图)
输出: D:/tmp_timing/selection_analysis.json
"""
import json
from pathlib import Path

import numpy as np

SR = Path(r'D:/models/sweep_test_out/sweep_results.json')
REP = Path(r'D:/tmp_timing/per_image_report.json')
F2K_VAL = Path(r'D:/tmp_timing/sweep_f2k.json')

OLD_VAL = {
    'old_s3000': 'cnn_val_s3k', 'old_s5000': 'cnn_val_s5k', 'old_s7000': 'cnn_val_s7000',
    'old_s8000': 'cnn_val_s8000', 'old_s9000': 'cnn_val_s9k', 'old_s10000': 'cnn_val_s10k',
}
S6K = 'old_s6000'


def main():
    res = json.load(open(SR))['results']
    by_name = {r['name']: r for r in res}
    rep = json.load(open(REP))['test_clipiqa']

    val_lp = {}
    if F2K_VAL.exists():
        for r in json.load(open(F2K_VAL)):
            stem = r['ckpt'].replace('.pt', '').replace('net_', '')
            val_lp['f2k_' + stem] = r['lpips']
    for name, d in OLD_VAL.items():
        p = Path(rf'D:/tmp_timing/{d}/val_result.json')
        if p.exists():
            val_lp[name] = json.load(open(p))['lpips']
    val_lp.setdefault(S6K, 0.3583)  # s6k 官方口径 val lpips

    names = [r['name'] for r in res if 'test_clipiqa' in r and r.get('n', 0) >= 100]
    print(f'[{len(names)}] 个 ckpt 参与选择', flush=True)
    # v2 ckpts: 用训练 eval_log 的 val3_lpips (按最近 step 近似) 注入 val lpips
    try:
        ev = json.load(open(r'D:/models/metric_net_v2/eval_log.json'))
        for n in names:
            if n.startswith('metric_net_v2_'):
                step = int(n.rsplit('s', 1)[1])
                best = min(ev, key=lambda x: abs(x['step'] - step))
                val_lp[n] = best['val3_lpips']
        print('[ok] v2 val lpips 已注入 (val3 近似)', flush=True)
    except Exception as e:
        print('[warn] v2 val lpips 注入失败:', e, flush=True)
    pi = {n: by_name[n]['per_image'] for n in names}
    keys = sorted(set(pi[S6K] if S6K in pi else next(iter(pi.values()))) & set(rep), key=lambda k: int(k.replace('case', '')))

    s6k_pi = {k: rep[k]['s6k'] for k in keys}  # 统一用 per_image_report 的 s6k 作参考
    s6k_mean = float(np.mean(list(s6k_pi.values())))

    # 逐图最优
    per_best_name, per_best_val = {}, {}
    per_best_safe_name, per_best_safe_val = {}, {}
    safe = {n for n in names if val_lp.get(n) is None or val_lp.get(n, 99) <= val_lp[S6K] + 0.005}
    print(f'safe ckpt 池 (val lpips <= s6k+0.005 或未知): {len(safe)}/{len(names)}', flush=True)
    names_set = set(names)
    for k in keys:
        best_v, best_n = s6k_pi[k], S6K
        for n in names:
            v = pi[n].get(k, {}).get('clipiqa')
            if v is not None and v > best_v:
                best_v, best_n = v, n
        per_best_name[k], per_best_val[k] = best_n, best_v
        # 安全池版本
        best_v2, best_n2 = s6k_pi[k], S6K
        for n in names_set & safe:
            v = pi[n].get(k, {}).get('clipiqa')
            if v is not None and v > best_v2:
                best_v2, best_n2 = v, n
        per_best_safe_name[k], per_best_safe_val[k] = best_n2, best_v2

    def sel_mean(pb):
        return float(np.mean(list(pb.values())))

    print(f's6k 基线: {s6k_mean:.4f}', flush=True)
    print(f'逐图最优天花板 (全池): {sel_mean(per_best_val):.4f} (Δ {sel_mean(per_best_val)-s6k_mean:+.4f})', flush=True)
    print(f'逐图最优天花板 (安全池): {sel_mean(per_best_safe_val):.4f} (Δ {sel_mean(per_best_safe_val)-s6k_mean:+.4f})', flush=True)

    # 带边际策略
    from collections import Counter
    print('\n===== 边际策略 (全池) =====', flush=True)
    strategies = {}
    for M in [0.005, 0.01, 0.015, 0.02, 0.03]:
        chosen = {}
        for k in keys:
            if per_best_val[k] - s6k_pi[k] > M:
                chosen[k] = per_best_name[k]
            else:
                chosen[k] = S6K
        mean = float(np.mean([per_best_val[k] if chosen[k] != S6K else s6k_pi[k] for k in keys]))
        n_swap = sum(1 for k in keys if chosen[k] != S6K)
        cnt = Counter(chosen.values())
        strategies[M] = {'mean': round(mean, 4), 'delta': round(mean - s6k_mean, 4), 'n_swap': n_swap,
                         'top_choices': cnt.most_common(4)}
        print(f'  M={M:.3f}: mean={mean:.4f} Δ{mean-s6k_mean:+.4f} 换图 {n_swap}/100 选源 {cnt.most_common(3)}', flush=True)

    print('\n===== 边际策略 (安全池) =====', flush=True)
    strategies_safe = {}
    for M in [0.005, 0.01, 0.015, 0.02, 0.03]:
        chosen = {}
        for k in keys:
            if per_best_safe_val[k] - s6k_pi[k] > M:
                chosen[k] = per_best_safe_name[k]
            else:
                chosen[k] = S6K
        mean = float(np.mean([per_best_safe_val[k] if chosen[k] != S6K else s6k_pi[k] for k in keys]))
        n_swap = sum(1 for k in keys if chosen[k] != S6K)
        strategies_safe[M] = {'mean': round(mean, 4), 'delta': round(mean - s6k_mean, 4), 'n_swap': n_swap}
        print(f'  M={M:.3f}: mean={mean:.4f} Δ{mean-s6k_mean:+.4f} 换图 {n_swap}/100', flush=True)

    # 导出推荐策略的逐图选择 (装配用): 安全池 M=0.02 与全池 M=0.02
    def build_choice(pb_name, pb_val, M):
        return {k: (pb_name[k] if pb_val[k] - s6k_pi[k] > M else S6K) for k in keys}
    out = {'s6k_mean': s6k_mean,
           'ceiling_all': {'mean': sel_mean(per_best_val), 'per_image': per_best_name},
           'ceiling_safe': {'mean': sel_mean(per_best_safe_val), 'per_image': per_best_safe_name},
           'choice_safe_M0.02': build_choice(per_best_safe_name, per_best_safe_val, 0.02),
           'choice_all_M0.02': build_choice(per_best_name, per_best_val, 0.02),
           'strategies': strategies, 'strategies_safe': strategies_safe,
           'val_lpips_used': {k: round(v, 4) for k, v in val_lp.items() if k in names}}
    json.dump(out, open(r'D:/tmp_timing/selection_analysis.json', 'w'), indent=1)
    print('\n[saved] D:/tmp_timing/selection_analysis.json', flush=True)


if __name__ == '__main__':
    main()
