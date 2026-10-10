# -*- coding: utf-8 -*-
"""提交图质检 (2026-09-08): 100 张 jpg 完整性/尺寸/统计异常检查
用法: python qc_test100.py <output_dir>
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    outdir = Path(sys.argv[1])
    jpgs = sorted(outdir.glob('case*.jpg'))
    print(f'检查 {len(jpgs)} 张: {outdir}')
    issues = []
    stats = []
    for j in jpgs:
        try:
            img = Image.open(j)
            img.load()
            if img.size not in ((3072, 4096), (4096, 3072), (3072, 3072)):
                issues.append(f'{j.name}: 尺寸异常 {img.size}')
            a = np.asarray(img.convert('L'), dtype=np.float32)
            stats.append((j.name, a.mean(), a.std()))
        except Exception as e:
            issues.append(f'{j.name}: 无法打开 {e}')
    means = [s[1] for s in stats]
    stds = [s[2] for s in stats]
    if stats:
        m_mean = float(np.mean(means))
        m_std = float(np.mean(stds))
        print(f'亮度均值分布: {min(means):.1f}-{max(means):.1f} (均 {m_mean:.1f})')
        print(f'对比度(std)分布: {min(stds):.1f}-{max(stds):.1f} (均 {m_std:.1f})')
        # 异常: 全黑/全白/过曝/低对比
        for name, mn, sd in stats:
            if mn < 5 or mn > 250:
                issues.append(f'{name}: 亮度极端 {mn:.1f}')
            if sd < 3:
                issues.append(f'{name}: 疑似纯色/坏图 std={sd:.1f}')
    print(f'问题: {len(issues)}')
    for x in issues:
        print(' ', x)
    print('QC_PASS' if not issues else 'QC_ISSUES')


if __name__ == '__main__':
    main()
