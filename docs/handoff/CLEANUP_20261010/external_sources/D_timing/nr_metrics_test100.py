# -*- coding: utf-8 -*-
"""测试集无参指标对比 (2026-09-08): 4 包 100 张渲染的 clipiqa/musiq/niqe
无参指标不需要 GT -> 可直接在测试集上评估官方无参项 (4/8 指标) 的胜势
用法: python nr_metrics_test100.py
"""
import json
import sys
import time
from pathlib import Path

import torch

OUT_ROOT = Path(r'D:/models')
PKGS = {
    'M24': OUT_ROOT / 'final_compliant/output_dir',
    'm0s1': OUT_ROOT / 'final_compliant_14b/output_dir',
    'g3w18s1': OUT_ROOT / 'final_compliant_g3w18s1/output_dir',
    'm0s7': OUT_ROOT / 'final_compliant_m0s7/output_dir',
}
RES = Path(r'D:/tmp_timing/nr_test100_results.json')
SAMPLE = 30  # 抽 30 张 (速度与统计稳定的折中)


def main():
    import pyiqa
    from PIL import Image
    import numpy as np
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = {}
    for name in ('clipiqa', 'musiq', 'niqe'):
        metrics[name] = pyiqa.create_metric(name, device=dev)
        print(f'[metric] {name} loaded', flush=True)
    results = {}
    for pkg, d in PKGS.items():
        jpgs = sorted(d.glob('*.jpg'))[:SAMPLE]
        if not jpgs:
            print(f'[skip] {pkg}: 无图')
            continue
        vals = {k: [] for k in metrics}
        t0 = time.time()
        for j in jpgs:
            img = Image.open(j).convert('RGB')
            t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            t = t.to(dev)
            for k, m in metrics.items():
                with torch.no_grad():
                    v = m(t)
                vals[k].append(float(v.item() if hasattr(v, 'item') else v))
        results[pkg] = {k: round(float(sum(v) / len(v)), 4) for k, v in vals.items()}
        print(f'[{pkg}] {len(jpgs)} 张 ({time.time()-t0:.0f}s): ' +
              ' '.join(f'{k}={results[pkg][k]:.4f}' for k in metrics), flush=True)
    RES.write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('\n=== 汇总 (测试集无参指标, 抽 30 张) ===')
    print(f'{"包":<10}{"clipiqa":<10}{"musiq":<10}{"niqe":<10}')
    for pkg, r in results.items():
        print(f'{pkg:<10}{r["clipiqa"]:<10.4f}{r["musiq"]:<10.4f}{r["niqe"]:<10.4f}')
    print(f'\n已存: {RES}')


if __name__ == '__main__':
    main()
