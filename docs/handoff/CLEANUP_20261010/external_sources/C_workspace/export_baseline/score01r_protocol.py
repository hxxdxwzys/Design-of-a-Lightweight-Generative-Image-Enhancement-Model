# -*- coding: utf-8 -*-
"""SCORE-01R 预登记协议（纯函数，CPU 可测）：块顺序、统计、漂移与门槛判定。

§0H 规定：
  - 两轮分别为 ABBA（A 基线在两端）与 BAAB（A 在中间）；
  - 每 block 每输入 2 次预热 + 10 次计时；进程首调/加载单列；
  - 以该轮两个 A block 汇总作为配对参考，所有 B block 纳入；
  - 每轮前后 A 基线三输入中位数均值漂移 >5% ⇒ 该轮 INVALID（非候选 FAIL）；
  - 门槛：两轮有效且每轮三输入中位数均值下降 ≥10%、各输入 p90 恶化 ≤5%。
"""
from __future__ import annotations

import statistics

BLOCK_ORDERS = {1: 'ABBA', 2: 'BAAB'}
WARMUP = 2
TIMED = 10
MEDIAN_IMPROVEMENT_MIN = 0.10
P90_WORSEN_MAX = 0.05
DRIFT_INVALID = 0.05
INPUTS = ('c1_512', 'c7_512', 'c25_512')


def block_order(round_index):
    if round_index not in BLOCK_ORDERS:
        raise ValueError('round_index 只能是 1 或 2，得到 %r' % (round_index,))
    return list(BLOCK_ORDERS[round_index])


def p90(values):
    """线性插值 p90（与 numpy.percentile(v, 90) 同式，免依赖）。"""
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError('空样本')
    if len(xs) == 1:
        return xs[0]
    rank = 0.9 * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def stats(values):
    xs = [float(v) for v in values]
    if not xs:
        raise ValueError('空样本')
    return {'n': len(xs), 'median': statistics.median(xs), 'p90': p90(xs),
            'mean': sum(xs) / len(xs), 'min': min(xs), 'max': max(xs)}


def side_values(blocks, label, name):
    """把该轮所有同标签 block 的某输入原始读数汇总（不挑最好）。"""
    out = []
    for b in blocks:
        if b['label'] != label:
            continue
        out.extend(float(v) for v in b['inputs'][name]['timed_ms'])
    return out


def a_block_median_means(blocks):
    """每个 A block 的三输入中位数均值（用于漂移判定，按 block 顺序）。"""
    out = []
    for b in blocks:
        if b['label'] != 'A':
            continue
        meds = [statistics.median(b['inputs'][n]['timed_ms']) for n in INPUTS]
        out.append(sum(meds) / len(meds))
    return out


def drift_ratio(blocks):
    """前后 A block 的三输入中位数均值相对漂移（|后-前|/前）。"""
    mm = a_block_median_means(blocks)
    if len(mm) < 2:
        return None
    return abs(mm[-1] - mm[0]) / mm[0]


def round_metrics(blocks, inputs=INPUTS):
    """按协议汇总：A/B 各输入中位数与 p90、改善比、p90 恶化比。"""
    res = {'inputs': {}, 'median_avg_A': None, 'median_avg_B': None,
           'median_improvement': None, 'drift': None, 'verdict': None}
    med_a, med_b = [], []
    ok_median = True
    ok_p90 = True
    for name in inputs:
        av = side_values(blocks, 'A', name)
        bv = side_values(blocks, 'B', name)
        sa, sb = stats(av), stats(bv)
        imp = (sa['median'] - sb['median']) / sa['median']
        p90_rel = (sb['p90'] - sa['p90']) / sa['p90']
        med_a.append(sa['median'])
        med_b.append(sb['median'])
        ok_median = ok_median and imp >= MEDIAN_IMPROVEMENT_MIN
        ok_p90 = ok_p90 and p90_rel <= P90_WORSEN_MAX
        res['inputs'][name] = {'A': sa, 'B': sb, 'median_improvement': imp,
                               'p90_rel_change': p90_rel, 'median_ok': imp >= MEDIAN_IMPROVEMENT_MIN,
                               'p90_ok': p90_rel <= P90_WORSEN_MAX}
    res['median_avg_A'] = sum(med_a) / len(med_a)
    res['median_avg_B'] = sum(med_b) / len(med_b)
    res['median_improvement'] = ((res['median_avg_A'] - res['median_avg_B'])
                                 / res['median_avg_A'])
    res['drift'] = drift_ratio(blocks)
    res['median_ok'] = bool(ok_median)
    res['p90_ok'] = bool(ok_p90)
    if res['drift'] is None:
        res['verdict'] = 'MALFORMED'
    elif res['drift'] > DRIFT_INVALID:
        res['verdict'] = 'INVALID'
    else:
        res['verdict'] = ('PASS' if (ok_median and ok_p90) else 'NO_GAIN')
    return res


def final_verdict(rounds):
    """rounds: [round_metrics, ...]（有效轮）。返回 §0H 四分类之一。"""
    valid = [r for r in rounds if r.get('verdict') in ('PASS', 'NO_GAIN')]
    if len(valid) < 2:
        return {'verdict': 'INCONCLUSIVE',
                'reason': '有效轮不足 2（漂移超限或轮次缺失）',
                'n_valid': len(valid), 'n_total': len(rounds)}
    passed = all(r['verdict'] == 'PASS' for r in valid[:2])
    return {'verdict': 'VALID_PASS' if passed else 'VALID_NO_GAIN',
            'reason': '两轮均达门槛' if passed else '两轮均未达门槛（无收益）',
            'rounds': [{'verdict': r['verdict'], 'median_improvement': r['median_improvement'],
                        'p90_ok': r['p90_ok'], 'drift': r['drift']} for r in valid[:2]]}
