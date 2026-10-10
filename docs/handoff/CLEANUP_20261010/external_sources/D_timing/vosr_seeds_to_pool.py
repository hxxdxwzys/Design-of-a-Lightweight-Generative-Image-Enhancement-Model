# -*- coding: utf-8 -*-
"""多种子 VOSR 输出入池: 转 jpg -> 逐图指标 -> 并入 sweep_results.json (entry: vosr_seed{seed})
用法: D:/Python/python.exe vosr_seeds_to_pool.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, r'D:/tmp_timing')
import pyiqa  # noqa: E402
from sweep_test_all import metrics_for_dir, BASE_OUT, RESULTS_JSON  # noqa: E402

SEEDS = [0, 123, 7, 999, 2024, 111, 222, 333, 444, 555]
SRC = Path(r'C:/models/finals/preset')


def main():
    device = 'cuda'
    clipiqa_m = pyiqa.create_metric('clipiqa', device=device)
    niqe_m = pyiqa.create_metric('niqe', device=device)
    d = json.load(open(RESULTS_JSON))
    results = [r for r in d.get('results', []) if not r['name'].startswith('vosr_seed')]
    for seed in SEEDS:
        name = f'vosr_seed{seed}'
        src_dir = SRC / f'vosr_seed{seed}_test'
        dst = BASE_OUT / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        n = 0
        for p in sorted(src_dir.glob('*.png')):
            k = p.stem.replace('_lq', '')
            Image.open(p).convert('RGB').save(dst / f'{k}.jpg', quality=98)
            n += 1
        print(f'[convert] {name}: {n} jpgs', flush=True)
        if n < 100:
            print(f'[warn] {name} 不足 100, 跳过', flush=True)
            continue
        m = metrics_for_dir(dst, clipiqa_m, niqe_m, device)
        results.append({'name': name, **m})
        json.dump({'results': results, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(RESULTS_JSON, 'w'), indent=1)
        print(f'[ok] {name}: clipiqa={m["test_clipiqa"]:.4f} niqe={m["test_niqe"]:.2f}, entries={len(results)}', flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
