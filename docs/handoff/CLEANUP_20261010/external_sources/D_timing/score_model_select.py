# -*- coding: utf-8 -*-
"""官方分模型选择 (2026-09-12): 用 5 个已得官方分的资产 + 同口径测试集指标, 反推"官方到底认哪个指标"。

口径: 全部为测试集 100 张里的 sorted()[:30] 前缀 (nr_test100_results.json / nr_extra_0911.json),
官方分为同资产多次提交的均值 (榜单取最好, 但拟合用均值更稳)。

用法: python score_model_select.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 资产 -> (测试集指标[30张同口径], 官方分[多次提交均值], 提交次数)
ASSETS = {
    'M24':        ({'niqe': 5.2559, 'musiq': 46.1257, 'clipiqa': 0.9023}, [43.6951, 43.6731, 43.5844]),
    'm0s1':       ({'niqe': 6.0195, 'musiq': 38.6290, 'clipiqa': 0.8745}, [34.0989, 34.1056]),
    'm0s7_s097':  ({'niqe': 6.3431, 'musiq': 37.8934, 'clipiqa': 0.8766}, [31.3942, 30.9363]),
    'g3w18s1':    ({'niqe': 6.2384, 'musiq': 37.7513, 'clipiqa': 0.8612}, [31.8185]),
    'm0s7_sc10':  ({'niqe': 6.3695, 'musiq': 37.8679, 'clipiqa': 0.8790}, [30.5914, 30.6253]),
}
# 官方噪声标尺 (同图重交配对差): 0.007~0.458
NOISE = 0.5


def main():
    names = list(ASSETS)
    y = np.array([np.mean(ASSETS[n][1]) for n in names])
    print('资产 / 官方分(均值, 次数):')
    for n in names:
        print(f'  {n:<11}{y[names.index(n)]:>9.4f}  ({" ".join(f"{s:.4f}" for s in ASSETS[n][1])})')

    # ---- 单指标线性拟合 ----
    print(f'\n=== 单指标线性拟合 (n={len(names)}; rmse 需 << {NOISE} 才算"能解释") ===')
    rows = []
    for k in ('niqe', 'musiq', 'clipiqa'):
        x = np.array([ASSETS[n][0][k] for n in names])
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        r = y - pred
        rows.append((k, b, a, float(np.sqrt((r ** 2).mean())), float(np.abs(r).max())))
        print(f'  {k:<9} Score = {a:8.3f} {b:+8.3f}·{k}   rmse={rows[-1][3]:6.3f}  max|res|={rows[-1][4]:6.3f}')
        print(f'           残差: ' + '  '.join(f'{n}={v:+.2f}' for n, v in zip(names, r)))

    # ---- 两指标拟合 ----
    print('\n=== 两指标最小二乘 (3 参数 / 5 点) ===')
    best = []
    for k1, k2 in itertools.combinations(('niqe', 'musiq', 'clipiqa'), 2):
        X = np.column_stack([np.array([ASSETS[n][0][k] for n in names]) for k in (k1, k2)]
                            + [np.ones(len(names))])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        r = y - pred
        rmse = float(np.sqrt((r ** 2).mean()))
        print(f'  {k1}+{k2:<8} coef={np.round(coef, 4)}  rmse={rmse:6.3f}  '
              f'max|res|={np.abs(r).max():6.3f}')
        print(f'             残差: ' + '  '.join(f'{n}={v:+.2f}' for n, v in zip(names, r)))
        best.append((rmse, (k1, k2), coef))

    # ---- 判别力检查: 各指标在"近同位对"上的符号 ----
    print('\n=== 判别力检查 (官方分差 vs 各指标预测符号) ===')
    for i, j in itertools.combinations(range(len(names)), 2):
        dy = y[i] - y[j]
        if abs(dy) < 1.0:
            continue
        line = f'  {names[i]} - {names[j]}: 官方 Δ={dy:+.3f}'
        for k in ('niqe', 'musiq', 'clipiqa'):
            dx = ASSETS[names[i]][0][k] - ASSETS[names[j]][0][k]
            line += f' | {k} Δ={dx:+.4f}'
        print(line)

    # ---- 顶部外推: 对手 60-81 需要什么 ----
    print('\n=== 关键外推: 用单指标模型反推对手 60/81 分所需的指标值 ===')
    for k, b, a, rmse, mx in rows:
        for target in (60.0, 81.0):
            need = (target - a) / b
            print(f'  {k:<9} 若 Score={target:>4.0f} -> {k} = {need:8.3f}'
                  f'   (我们最好 {ASSETS["M24"][0][k]:.4f})')
        print()

    # ---- 曲线形状判别 (val, 方向性) ----
    print('=== 阶段 1 四点曲线: 各指标的 argmax (val 3 张, 仅看形状) ===')
    curve = {  # scale -> val 指标
        0.0: {'niqe': 4.749, 'clipiqa': 0.615, 'musiq': 42.14, 'maniqa': 0.445,
              'lpips': 0.342, 'dists': 0.2105, 'psnr': 24.389, 'ssim': 0.7453},
        0.4: {'niqe': 4.4586, 'clipiqa': 0.787, 'musiq': 45.87, 'maniqa': 0.476,
              'lpips': 0.361, 'dists': 0.2500, 'psnr': 24.471, 'ssim': 0.7440},
        0.8: {'niqe': 5.172, 'clipiqa': 0.919, 'musiq': 48.14, 'maniqa': 0.514,
              'lpips': 0.385, 'dists': 0.2960, 'psnr': 24.604, 'ssim': 0.7417},
        1.0: {'niqe': 5.666, 'clipiqa': 0.938, 'musiq': 45.57, 'maniqa': 0.504,
              'lpips': 0.396, 'dists': 0.3006, 'psnr': 24.711, 'ssim': 0.7394},
    }
    better = {'niqe': 'min', 'clipiqa': 'max', 'musiq': 'max', 'maniqa': 'max',
              'lpips': 'min', 'dists': 'min', 'psnr': 'max', 'ssim': 'max'}
    for k, d in better.items():
        vals = {s: curve[s][k] for s in curve}
        best_s = min(vals, key=vals.get) if d == 'min' else max(vals, key=vals.get)
        seq = ' > '.join(f'{s}:{vals[s]:.4g}' for s in sorted(vals))
        print(f'  {k:<8}({d}) argmax@sc{best_s}   {seq}')
    print('\n  ➜ 若官方分 argmax = sc0.4 -> niqe 家族; sc0.0 -> lpips/dists; '
          'sc0.8 -> musiq/maniqa; sc1.0 -> clipiqa')


if __name__ == '__main__':
    main()
