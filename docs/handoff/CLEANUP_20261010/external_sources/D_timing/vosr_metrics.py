# -*- coding: utf-8 -*-
"""VOSR 测试集输出逐图指标 -> 并入 sweep_results.json (entry: vosr_05b_os)
前提: vosr_to_pool.py 已把 jpg 放进 D:/models/sweep_test_out/vosr_05b_os/
用法: D:/Python/python.exe vosr_metrics.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, r'D:/tmp_timing')
import pyiqa  # noqa: E402
from sweep_test_all import metrics_for_dir, BASE_OUT, RESULTS_JSON  # noqa: E402


def main():
    device = 'cuda'
    d = json.load(open(RESULTS_JSON))
    results = [r for r in d.get('results', []) if r['name'] != 'vosr_05b_os']
    clipiqa_m = pyiqa.create_metric('clipiqa', device=device)
    niqe_m = pyiqa.create_metric('niqe', device=device)
    m = metrics_for_dir(BASE_OUT / 'vosr_05b_os', clipiqa_m, niqe_m, device)
    results.append({'name': 'vosr_05b_os', **m})
    json.dump({'results': results, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(RESULTS_JSON, 'w'), indent=1)
    print(f'[ok] vosr_05b_os: clipiqa={m["test_clipiqa"]:.4f} niqe={m["test_niqe"]:.2f}, entries={len(results)}', flush=True)


if __name__ == '__main__':
    main()
