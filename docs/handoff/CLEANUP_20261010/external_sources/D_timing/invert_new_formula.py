# -*- coding: utf-8 -*-
"""新世界公式反演 v2 (2026-09-11): 用成对完整特征拟合, 缺失值自动跳过"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np

A = {
    'M24':       dict(score=43.6841, clip=0.9008, musiq=46.1257, niqe=5.2559,
                      lpips=0.3852, dists=0.3299, fam='0.5B'),
    'm0s1':      dict(score=34.1022, clip=0.8780, musiq=38.6290, niqe=6.0195,
                      lpips=0.3571, dists=0.3414, fam='1.4B'),
    'g3w18s1':   dict(score=31.8185, clip=0.8612, musiq=37.7513, niqe=6.2384,
                      lpips=0.3618, dists=0.3192, fam='1.4B'),
    'm0s7_s097': dict(score=31.1653, clip=0.8838, musiq=None, niqe=None,
                      lpips=0.3545, dists=None, fam='1.4B'),
    'm0s7_sc10': dict(score=30.6084, clip=0.8790, musiq=37.8679, niqe=6.3695,
                      lpips=0.3571, dists=None, fam='1.4B'),
}


def fit(feats, pool=None, label=''):
    ks = [k for k, v in A.items()
          if (pool is None or k in pool) and all(v[f] is not None for f in feats)]
    if len(ks) < len(feats) + 1:
        print(f'  [{label}] 可用点不足 ({len(ks)} < {len(feats)+1}) -> 跳过')
        return None, ks
    X = np.array([[A[k][f] for f in feats] for k in ks])
    y = np.array([A[k]['score'] for k in ks])
    M = np.concatenate([np.ones((len(ks), 1)), X], axis=1)
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    pred = M @ coef
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    print(f'  [{label}] n={len(ks)} feats={feats}')
    print('     ' + '  '.join(f'{n}={c:+.3f}' for n, c in zip(['C'] + feats, coef)) + f'   rmse={rmse:.3f}')
    for k, p in zip(ks, pred):
        print(f'       {k:<11} 实测 {A[k]["score"]:7.3f}  拟合 {p:7.3f}  残差 {A[k]["score"]-p:+6.3f}')
    return (coef, feats, rmse), ks


print('=' * 96)
print('① 1.4B 家族内部（同速→差异只来自质量; 只用四个 1.4B 中特征完整的点）')
print('=' * 96)
r1, _ = fit(['musiq', 'niqe'], pool={'m0s1', 'g3w18s1', 'm0s7_sc10'}, label='musiq+niqe')
fit(['musiq'], pool={'m0s1', 'g3w18s1', 'm0s7_sc10'}, label='musiq only')
fit(['niqe'], pool={'m0s1', 'g3w18s1', 'm0s7_sc10'}, label='niqe only')
fit(['lpips'], pool={'m0s1', 'g3w18s1', 'm0s7_sc10', 'm0s7_s097'}, label='lpips only(4点)')
fit(['clip'], pool={'m0s1', 'g3w18s1', 'm0s7_sc10', 'm0s7_s097'}, label='clipiqa only(4点)')

print()
print('=' * 96)
print('② 用 1.4B 拟合出的模型外推 M24（检验是否存在未识别项）')
print('=' * 96)
if r1:
    coef, feats, _ = r1
    m = A['M24']
    p = coef[0] + sum(c * m[f] for c, f in zip(coef[1:], feats))
    print(f'  M24 实测 {m["score"]:.3f} | 模型外推 {p:.3f} | 残差 {m["score"]-p:+.3f}')
    print(f'  → M24 相对"质量模型"的额外偏差 {m["score"]-p:+.1f} 分（速度项/musiq 饱和/本地测不到的指标）')

print()
print('=' * 96)
print('③ 关键结论')
print('=' * 96)
print('  排序: M24 43.68 >> m0s1 34.10 > g3w18s1 31.82 > m0s7_s097 31.17 > m0s7_sc10 30.61')
print('  · lpips 与分数【反相关】→ 我们一年优化的主轴在新公式里不被奖励')
print('  · 1.4B 内部差异可由 musiq↑ / niqe↓ 解释')
print('  · M24 的领先远超质量模型外推 → 存在未识别项（速度 或 本地测不到的指标）')
