# -*- coding: utf-8 -*-
"""恢复完整选择池数据: 
  1) 合并 2057 备份的 14 条 v2b 逐图数据 (探针不重跑它们, 会被丢);
  2) 补算 5 个 old_* 目录的逐图指标 (ckpt 已删但渲染目录在);
  3) 全部合并写回 sweep_results.json (不改动已有条目, 按 name 去重, 已有优先)。
前提: 探针 (sweep_v2b_early.log [all done]) 已结束。
用法: D:/Python/python.exe metrics_old_dirs.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, r'D:/tmp_timing')
import pyiqa  # noqa: E402
from sweep_test_all import metrics_for_dir, BASE_OUT, RESULTS_JSON  # noqa: E402

BACKUP = BASE_OUT / 'sweep_results.backup_20260815_2057.json'
OLD_DIRS = {
    'old_s5000': BASE_OUT / 'old_s5000',
    'old_s7000': BASE_OUT / 'old_s7000',
    'old_s8000': BASE_OUT / 'old_s8000',
    'old_s9000': BASE_OUT / 'old_s9000',
    'old_s10000': BASE_OUT / 'old_s10000',
}


def main():
    device = 'cuda'
    d = json.load(open(RESULTS_JSON))
    results = list(d.get('results', []))
    by_name = {r['name']: r for r in results}
    added = 0

    # 1) 合并备份的 v2b 条目
    if BACKUP.exists():
        b = json.load(open(BACKUP))
        for r in b.get('results', []):
            if r['name'] not in by_name:
                results.append(r)
                by_name[r['name']] = r
                added += 1
        print(f'[merge] 备份 v2b 条目并入 {added} 条', flush=True)

    # 2) 补算 5 个 old_* 目录
    clipiqa_m = pyiqa.create_metric('clipiqa', device=device)
    niqe_m = pyiqa.create_metric('niqe', device=device)
    for name, dpath in OLD_DIRS.items():
        if not dpath.is_dir() or len(list(dpath.glob('*.jpg'))) < 100:
            print(f'[skip] {name} 目录缺失', flush=True)
            continue
        m = metrics_for_dir(dpath, clipiqa_m, niqe_m, device)
        rec = {'name': name, **m}
        if name in by_name:
            by_name[name].update(rec)
        else:
            results.append(rec)
            by_name[name] = rec
        with open(RESULTS_JSON, 'w') as f:
            json.dump({'results': results, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=1)
        print(f'[ok] {name}: clipiqa={m["test_clipiqa"]:.4f} niqe={m["test_niqe"]:.2f}', flush=True)

    with open(RESULTS_JSON, 'w') as f:
        json.dump({'results': results, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=1)
    print(f'[done] total entries={len(results)}', flush=True)


if __name__ == '__main__':
    main()
