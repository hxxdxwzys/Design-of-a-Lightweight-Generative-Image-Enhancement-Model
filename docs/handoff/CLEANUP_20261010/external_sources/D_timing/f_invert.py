# -*- coding: utf-8 -*-
"""F' 公式反演器 (2026-09-09 版) — 数据源统一到 asset_profiles.json, 避免与 discriminate.py 分叉。

用法:
  python f_invert.py            # 用已记录的官方分做最小二乘/假设拟合
  (记录官方分: python discriminate.py --record m0s1=171.3)

候选公式族:
  H1: clip 饱和型  Score = C + a*max(0, clip_test - T) + b*lpips      (T 网格)
  H2: 纯 lpips 型  Score = C + b*lpips
  H3: 线性全指标   Score = C + sum(w_i * m_i)                          (需 >=5 点)
  H4: lpips + dists
输出: 各假设的残差/参数 + 判别下一发的最优资产。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROFILES = Path(r'D:/tmp_timing/asset_profiles.json')


def load_data():
    d = json.loads(PROFILES.read_text(encoding='utf-8'))
    pts = []
    for name, a in d['assets'].items():
        off = d['official_scores'].get(name)
        if off is None:
            continue
        clip_test = (a['test_nr_100']['clipiqa'] if a.get('test_nr_100')
                     else a['test_nr_30']['clipiqa'])
        pts.append({
            'name': name, 'official': float(off),
            'm': {
                'psnr': a['val']['psnr'], 'ssim': a['val']['ssim'],
                'lpips': a['val']['lpips'], 'dists': a['val']['dists'],
                'clipiqa': clip_test, 'musiq': a['test_nr_30']['musiq'],
                'niqe': a['test_nr_30']['niqe'], 'val_clip': a['val']['clipiqa'],
            },
        })
    return pts


def fit_linear(points, feats):
    """最小二乘: score = C + sum(w*f); 返回 (coef, rmse, 系数表)"""
    import numpy as np
    X = np.array([[p['m'][f] if p['m'].get(f) is not None else np.nan for f in feats]
                  for p in points], float)
    y = np.array([p['official'] for p in points], float)
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[ok], y[ok]
    if len(X) < len(feats) + 1:
        return None, None, None
    A = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rmse = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
    return coef, rmse, list(zip(['C'] + feats, coef))


def main():
    pts = load_data()
    print(f'=== F\' 反演器: {len(pts)} 个官方点 ===')
    for p in pts:
        print(f'  {p["name"]:<10} official={p["official"]:.4f}  '
              f'lpips={p["m"]["lpips"]} clip_test={p["m"]["clipiqa"]} '
              f'dists={p["m"]["dists"]}')
    if len(pts) < 2:
        print('\n官方点不足 (需 >=2)。首发 M24 校准弹 -> 第 2 发 m0s1 判别。')
        print('先跑 python discriminate.py 看判别设计与情景预测。')
        return
    print()
    for name, feats in [('H2_纯lpips', ['lpips']),
                        ('H4_lpips+dists', ['lpips', 'dists']),
                        ('H1_线性clip+lpips', ['clipiqa', 'lpips']),
                        ('H3_全指标', ['psnr', 'ssim', 'lpips', 'dists', 'clipiqa'])]:
        coef, rmse, tbl = fit_linear(pts, feats)
        if coef is None:
            print(f'{name:<18} 点不足 ({len(feats)} 特征需 >= {len(feats)+1} 点)')
            continue
        print(f'{name:<18} rmse={rmse:7.4f}  ' +
              ' '.join(f'{k}={v:+.1f}' for k, v in tbl))
    print('\n[提示] 判别优先用 discriminate.py 的破衡权重法 (不依赖点数, 只依赖符号)。')


if __name__ == '__main__':
    main()
