# -*- coding: utf-8 -*-
"""多种子 VOSR 输出入池 v2 (安全合并语义): 只替换本次 SEEDS 列表内的条目, 保留其余
用法: D:/Python/python.exe vosr_seeds_to_pool2.py [--skip-existing]   (--skip-existing: 已入池且 n>=100 的种子跳过, 供断点续跑)
"""
import argparse
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

# 本次渲染的种子 (10->20 的新 10 个; 旧 10 个已入池, 不在本列表则保留不动)
SEEDS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
SRC = Path(r'C:/models/finals/preset')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    device = 'cuda'
    clipiqa_m = pyiqa.create_metric('clipiqa', device=device)
    niqe_m = pyiqa.create_metric('niqe', device=device)
    d = json.load(open(RESULTS_JSON))
    # 保留所有非本次列表内的条目 (含旧 vosr_seed 条目)
    results = [r for r in d.get('results', []) if r['name'] not in {f'vosr_seed{s}' for s in SEEDS}]
    for seed in SEEDS:
        name = f'vosr_seed{seed}'
        if args.skip_existing and any(r['name'] == name and r.get('n', 0) >= 100 for r in results):
            print(f'[skip] {name}: 已入池', flush=True)
            continue
        src_dir = SRC / f'vosr_seed{seed}_test'
        dst = BASE_OUT / name
        if not src_dir.exists():
            print(f'[skip] {name}: 源目录不存在', flush=True)
            continue
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
