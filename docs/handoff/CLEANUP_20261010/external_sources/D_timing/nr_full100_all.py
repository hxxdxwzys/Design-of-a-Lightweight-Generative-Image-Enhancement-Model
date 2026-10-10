# -*- coding: utf-8 -*-
"""全 100 张测试集无参指标 (4 包 x clipiqa/musiq/niqe, 2026-09-09)
背景: 30 张抽样已有 -> 升级到 100 张, 为判别序列 ③④ 与 "无参权重" 假设提供决定性数据。
断点续跑: 已算过的 (pkg, metric, case) 跳过。
用法: python nr_full100_all.py [--metrics clipiqa,musiq,niqe]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PKGS = {
    'M24': Path(r'D:/models/final_compliant/output_dir'),
    'm0s1': Path(r'D:/models/final_compliant_14b/output_dir'),
    'g3w18s1': Path(r'D:/models/final_compliant_g3w18s1/output_dir'),
    'm0s7': Path(r'D:/models/final_compliant_m0s7/output_dir'),
}
OUT = Path(r'D:/tmp_timing/nr_full100_all.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', default='clipiqa,musiq,niqe')
    args = ap.parse_args()
    metric_names = [m.strip() for m in args.metrics.split(',') if m.strip()]

    import numpy as np
    import torch
    import pyiqa
    from PIL import Image

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    metrics = {}
    for name in metric_names:
        metrics[name] = pyiqa.create_metric(name, device=dev)
        print(f'[metric] {name} ready', flush=True)

    for pkg, d in PKGS.items():
        jpgs = sorted(d.glob('*.jpg'))
        if not jpgs:
            print(f'[skip] {pkg}: 无图')
            continue
        res.setdefault(pkg, {})
        for name, m in metrics.items():
            res[pkg].setdefault(name, {})
            t0, done = time.time(), 0
            for j in jpgs:
                if j.stem in res[pkg][name]:
                    continue
                img = Image.open(j).convert('RGB')
                t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)
                t = t.permute(2, 0, 1).unsqueeze(0).to(dev)
                with torch.no_grad():
                    v = m(t)
                res[pkg][name][j.stem] = round(float(v.item() if hasattr(v, 'item') else v), 4)
                done += 1
                if done % 20 == 0:
                    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
                    print(f'  [{pkg}/{name}] {done}/{len(jpgs)} '
                          f'({time.time()-t0:.0f}s)', flush=True)
            OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
            vals = np.array(list(res[pkg][name].values()))
            print(f'[{pkg}/{name}] n={len(vals)} mean={vals.mean():.4f} '
                  f'std={vals.std():.4f} min={vals.min():.4f} max={vals.max():.4f} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    print('\n=== 汇总 (100 张测试集无参) ===')
    hdr = f'{"pkg":<10}' + ''.join(f'{m:<12}' for m in metric_names)
    print(hdr)
    for pkg in PKGS:
        if pkg not in res:
            continue
        row = f'{pkg:<10}'
        for m in metric_names:
            v = res[pkg].get(m, {})
            row += f'{np.mean(list(v.values())):<12.4f}' if v else f'{"-":<12}'
        print(row)
    if 'M24' in res:
        print('\n逐图胜负 vs M24:')
        for pkg in PKGS:
            if pkg == 'M24' or pkg not in res:
                continue
            for m in metric_names:
                a, b = res['M24'].get(m, {}), res[pkg].get(m, {})
                common = sorted(set(a) & set(b))
                if not common:
                    continue
                w = sum(1 for k in common if a[k] > b[k])
                print(f'  {pkg:<10} {m:<10} M24 胜 {w} / {pkg} 胜 {len(common)-w}')
    print(f'\n已存: {OUT}')


if __name__ == '__main__':
    main()
