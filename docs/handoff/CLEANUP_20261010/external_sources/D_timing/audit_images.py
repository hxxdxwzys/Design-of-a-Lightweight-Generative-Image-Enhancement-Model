# -*- coding: utf-8 -*-
"""提交图完整性/一致性审计 (2026-09-11)
① 尺寸：每张出图必须与对应 LQ 尺寸【完全一致】(方向/宽高错一张就是废图)
② 颜色：出图 vs LQ 的逐通道均值/标准差（检测系统性偏色/亮度漂移）
③ 数值：全黑/全白/极低方差等退化图
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from PIL import Image

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
PKGS = {
    'M24': Path(r'D:/models/final_compliant/output_dir'),
    'm0s1': Path(r'D:/models/final_compliant_14b/output_dir'),
    'g3w18s1': Path(r'D:/models/final_compliant_g3w18s1/output_dir'),
    'm0s7_sc10': Path(r'D:/models/final_compliant_m0s7/output_dir'),
    'm0s7_s097': Path(r'D:/models/wave2_m0s7_s097/output_dir'),
}
OUT = Path(r'D:/tmp_timing/audit_images.json')


def stats(im):
    a = np.asarray(im.convert('RGB'), np.float32)
    return a.mean(axis=(0, 1)), a.std(axis=(0, 1)), a.mean(), a.std()


def main():
    rep = {}
    lq_stats = {}
    for f in sorted(TEST.glob('*.jpg')):
        im = Image.open(f)
        m, s, gm, gs = stats(im)
        lq_stats[f.name] = {'size': im.size, 'mean': [round(float(x), 2) for x in m],
                            'std': [round(float(x), 2) for x in s],
                            'gmean': round(float(gm), 2), 'gstd': round(float(gs), 2)}
    print(f'LQ 测试集: {len(lq_stats)} 张, 尺寸集合 = '
          f'{sorted({v["size"] for v in lq_stats.values()})}')

    for tag, d in PKGS.items():
        if not d.exists():
            print(f'[skip] {tag}'); continue
        size_bad, black, lowvar = [], [], []
        dm, ds, cnt = np.zeros(3), np.zeros(3), 0
        for f in sorted(d.glob('*.jpg')):
            im = Image.open(f)
            lq = lq_stats.get(f.name)
            if lq and im.size != lq['size']:
                size_bad.append((f.name, im.size, lq['size']))
                continue
            m, s, gm, gs = stats(im)
            dm += m; ds += s; cnt += 1
            if gm < 5 or gm > 250:
                black.append((f.name, round(float(gm), 1)))
            if gs < 3:
                lowvar.append((f.name, round(float(gs), 2)))
        dm, ds = dm / max(cnt, 1), ds / max(cnt, 1)
        # 与 LQ 的平均差异
        lm = np.mean([v['mean'] for v in lq_stats.values()], axis=0)
        ls = np.mean([v['std'] for v in lq_stats.values()], axis=0)
        rep[tag] = {'n': cnt, 'size_mismatch': size_bad[:5], 'black': black[:5],
                    'lowvar': lowvar[:5],
                    'mean_shift': [round(float(x), 2) for x in (dm - lm)],
                    'std_ratio': [round(float(x), 3) for x in (ds / ls)]}
        print(f'\n[{tag}] 有效 {cnt}/100')
        print(f'  尺寸不符: {len(size_bad)}  {size_bad[:3]}')
        print(f'  全黑/全白: {len(black)}   低方差(<3): {len(lowvar)}')
        print(f'  均值漂移(RGB vs LQ): {rep[tag]["mean_shift"]}')
        print(f'  标准差比(RGB vs LQ): {rep[tag]["std_ratio"]}')

    OUT.write_text(json.dumps({'lq': lq_stats, 'pkgs': rep}, indent=1), encoding='utf-8')
    print(f'\n已存 {OUT}')


if __name__ == '__main__':
    main()
