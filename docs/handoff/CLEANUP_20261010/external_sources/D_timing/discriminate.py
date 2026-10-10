# -*- coding: utf-8 -*-
"""判别序列执行器 (2026-09-09) — 官方通道重开后每发提交前后跑一次。

用法:
  python discriminate.py                     # 打印判别设计 (Δ 向量 + 破衡权重 + 情景预测)
  python discriminate.py --record m0s1=171.3 # 记录官方分 -> 自动反演 + 出下一发建议
  python discriminate.py --record M24=163.0 --record m0s1=171.3
  python discriminate.py --report            # 只读当前已记录的官方分并给结论

设计原则 (见 NEXT_ROUND_PROMPT_20260909 / THEORETICAL_CEILING_FRAMEWORK):
  * 官方公式未知 -> 不做点估计, 只做 "破衡权重" (sign flip 临界权重) 与情景预测
  * lpips 权重固定为旧世界参考 -400/unit 作归一化基准; 其余权重以该基准表达
  * 有参项只能 val 测 (需 GT); 无参项用测试集直证 (100 张 clipiqa / 30 张 musiq+niqe)
  * 判读阈值 THRESH=6.0 分 (官方噪声 +-0.4~0.75 -> 6 分约 8sigma, 显著)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:  # Windows 控制台/管道默认 GBK -> 统一 UTF-8 输出
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROFILES = Path(r'D:/tmp_timing/asset_profiles.json')
LPIPS_W = -400.0          # 旧世界参考斜率 (归一化基准)
CLIP_W = 1326.0           # 旧世界 clip 斜率 (饱和线以下段)
THRESH = 6.0              # 显著判定阈值 (分)
DS_14B = -1.5             # 1.4B 相对 0.5B 的速度分劣势 (估 -1~2)


def load():
    return json.loads(PROFILES.read_text(encoding='utf-8'))


def save(d):
    PROFILES.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')


def feat(a, key, prefer100=True):
    """取资产特征: 优先 100 张测试集口径, 退化到 30 张 / val"""
    if key == 'lpips':
        return a['val']['lpips']
    if key == 'dists':
        return a['val']['dists']
    if key == 'psnr':
        return a['val']['psnr']
    if key == 'ssim':
        return a['val']['ssim']
    if key == 'test_clip':
        if prefer100 and a.get('test_nr_100'):
            return a['test_nr_100']['clipiqa']
        return a['test_nr_30']['clipiqa']
    if key == 'test_musiq':
        return a['test_nr_30']['musiq']
    if key == 'test_niqe':
        return a['test_nr_30']['niqe']
    if key == 'val_clip':
        return a['val']['clipiqa']
    raise KeyError(key)


def delta(assets, a_name, b_name, prefer100=True):
    """b - a 的特征差向量"""
    A, B = assets[a_name], assets[b_name]
    keys = ['lpips', 'dists', 'psnr', 'ssim', 'test_clip', 'test_musiq', 'test_niqe', 'val_clip']
    out = {}
    for k in keys:
        try:
            va, vb = feat(A, k, prefer100), feat(B, k, prefer100)
        except KeyError:
            va = vb = None
        out[k] = None if (va is None or vb is None) else round(vb - va, 5)
    out['speed'] = 0.0 if A['family'] == B['family'] else (
        DS_14B if B['family'] == '1.4B' else -DS_14B)
    return out


def clip_term(clip, T):
    return CLIP_W * max(0.0, clip - T)


def scenario(assets, a_name, b_name, T, w_clip=CLIP_W, w_lpips=LPIPS_W,
             w_dists=0.0, w_musiq=0.0, w_niqe=0.0):
    """预测 Δ(B-A) 分"""
    d = delta(assets, a_name, b_name)
    A, B = assets[a_name], assets[b_name]
    s = 0.0
    s += w_lpips * d['lpips']
    s += clip_term(feat(B, 'test_clip'), T) - clip_term(feat(A, 'test_clip'), T)
    if w_dists and d['dists'] is not None:
        s += w_dists * d['dists']
    if w_musiq:
        s += w_musiq * d['test_musiq']
    if w_niqe:
        s += w_niqe * d['test_niqe']
    s += d['speed']
    return s


def breakeven(assets, a_name, b_name, feature, w_extra=None, dS=None):
    """给定 lpips 权重 -400, 求使 Δ(B-A)=0 的 feature 权重 (线性, 无饱和)"""
    d = delta(assets, a_name, b_name)
    base = LPIPS_W * d['lpips'] + (d['speed'] if dS is None else dS)
    if w_extra:
        for f, w in w_extra.items():
            base += w * d[f]
    df = d.get(feature)
    if not df:
        return None
    return -base / df


def design(assets):
    print('=' * 78)
    print('判别设计 (官方分未知时): 资产对 Δ 向量  [B - A]')
    print('=' * 78)
    pairs = [('M24', 'm0s1'), ('m0s1', 'g3w18s1'), ('m0s1', 'm0s7'), ('M24', 'g3w18s1')]
    for a, b in pairs:
        d = delta(assets, a, b)
        print(f'\n[{a} -> {b}]')
        for k, v in d.items():
            print(f'   Δ{k:<11} {v:+.5f}' if v is not None else f'   Δ{k:<11} n/a')
        # 情景
        for T in (0.9008, 0.90, 0.878, 0.872):
            pred = scenario(assets, a, b, T)
            tag = 'lpips 主轴' if pred > THRESH else ('M24/clip 主轴' if pred < -THRESH else '不可判')
            print(f'   T={T:<6} w_clip={CLIP_W:.0f} -> Δ_pred = {pred:+7.2f}  [{tag}]')
        for f in ('test_clip', 'test_musiq', 'test_niqe', 'dists'):
            be = breakeven(assets, a, b, f)
            if be is not None:
                df = delta(assets, a, b)[f]
                direction = f'{b} 赢' if df > 0 else f'{b} 输'
                print(f'   破衡 w_{f:<10} = {be:+.1f}  '
                      f'(权重比该值更极端时 {direction}; 方向: Δ{f}={df:+.5f})')
    print('\n判读规则 (观察到的官方 Δ = score(B) - score(A)):')
    print(f'  Δ >= +{THRESH}  -> 公式认 lpips (clip 对两方均饱和) => 1.4B 主线复活')
    print(f'  |Δ| <  {THRESH}  -> 不可判/混合权重       => 补发 g3w18s1 (dists) 或 m0s7 (σ)')
    print(f'  Δ <= -{THRESH}  -> clip 未饱和段主导        => M24 终局, 停 1.4B 线')


def report(assets):
    sc = {k: v for k, v in assets['official_scores'].items()
          if not k.startswith('_') and v is not None}
    print('=' * 78)
    print(f'官方分记录: {len(sc)} 点  {sc if sc else "(空)"}')
    print('=' * 78)
    if len(sc) < 2:
        print('需要 >=2 个点才能做 Δ 判读。首发建议 M24 (校准弹)。')
        return
    if 'M24' in sc and 'm0s1' in sc:
        obs = sc['m0s1'] - sc['M24']
        print(f'\n[核心判别] Δ(m0s1 - M24) = {obs:+.2f}')
        best = None
        for T in (0.9008, 0.90, 0.878, 0.872):
            pred = scenario(assets, 'M24', 'm0s1', T)
            print(f'   T={T:<6} -> 预测 {pred:+7.2f}   (残差 {obs - pred:+7.2f})')
            if best is None or abs(obs - pred) < abs(obs - best[1]):
                best = (T, pred)
        print(f'  最佳拟合 T = {best[0]}')
        if obs >= THRESH:
            print('  => 判定: 公式认 lpips, clip 已饱和 -> 1.4B 主线复活。')
            print('     下一发: g3w18s1 (dists 判别) 或 m0s7 (σ/免 scale 备选)。')
        elif obs <= -THRESH:
            print('  => 判定: clip 未饱和段主导 -> M24 终局, 停止 1.4B 提交。')
            print('     下一发: 保留预算, 只做 M24 家族内改进/收手。')
        else:
            print('  => 判定: 不可判 (混合权重或噪声)。')
            print('     下一发: g3w18s1 (dists 轴) 三点反演, 或 m0s7 测 σ。')
    if 'g3w18s1' in sc and 'm0s1' in sc:
        obs = sc['g3w18s1'] - sc['m0s1']
        be = breakeven(assets, 'm0s1', 'g3w18s1', 'dists')
        print(f'\n[dists 判别] Δ(g3w18s1 - m0s1) = {obs:+.2f}; 破衡 w_dists = {be}')
        if be is not None:
            print(f'  => 若 |w_dists| 远小于 {abs(be):.1f}, 观测 Δ 应接近 '
                  f'{scenario(assets, "m0s1", "g3w18s1", 0.9008):+.2f} (lpips 主导)')
    if 'm0s7' in sc and 'm0s1' in sc:
        obs = sc['m0s7'] - sc['m0s1']
        print(f'\n[噪声估计] Δ(m0s7 - m0s1) = {obs:+.2f} (两资产近同位 -> 该值即 σ 量级)')


def main():
    d = load()
    args = sys.argv[1:]
    if not args:
        design(d['assets'])
        report(d)
        return
    if '--record' in args:
        for i, a in enumerate(args):
            if a == '--record' and i + 1 < len(args):
                k, _, v = args[i + 1].partition('=')
                if k in d['assets']:
                    d['official_scores'][k] = float(v)
                    print(f'[record] {k} = {v}')
                else:
                    print(f'[skip] 未知资产 {k}')
        d['budget']['used_new_world'] = sum(
            1 for k, v in d['official_scores'].items()
            if not k.startswith('_') and v is not None)
        save(d)
    report(load())


if __name__ == '__main__':
    main()
