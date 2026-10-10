# -*- coding: utf-8 -*-
"""探针 A 指标复核 (2026-09-12): 坐实/推翻 "sc0.4 测试集 niqe 大幅下降" 这一结论。
此前 probe_a_screen.json 写入的是 NaN (渲染器输出 PNG, 脚本 glob 了 *.jpg),
被引用的 4.0976 在磁盘上无任何文件支撑 -> 必须重测。

口径: 与 nr_test100_results.json 相同的 sorted(测试集)[:30] 前缀 30 张;
对照 M24 (final_compliant/output_dir, 已得官方 43.6951) 的同名文件, 逐图配对。

用法: python verify_probe_a_metrics.py [--wait-min 120]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
SC04 = Path(r'D:/models/final_compliant_probeA_sc04/output_dir')
M24 = Path(r'D:/models/final_compliant/output_dir')
OUT = Path(r'D:/tmp_timing/probe_a_verify.json')
N = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait-min', type=float, default=150)
    a = ap.parse_args()

    t0 = time.time()
    while len(list(SC04.glob('*.jpg'))) < 100:
        if time.time() - t0 > a.wait_min * 60:
            print(f'[超时] 渲染未完成, 现有 {len(list(SC04.glob("*.jpg")))} 张'); return
        time.sleep(60)
    print(f'[ok] sc0.4 渲染完成 ({len(list(SC04.glob("*.jpg")))} 张)', flush=True)

    names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:N]]
    pairs = [(n, SC04 / f'{n}.jpg', M24 / f'{n}.jpg') for n in names]
    miss = [n for n, p, q in pairs if not p.exists() or not q.exists()]
    if miss:
        print(f'[缺图] {miss[:5]} ... 共 {len(miss)}'); return

    import numpy as np
    import torch
    import pyiqa
    from PIL import Image
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    keys = ('niqe', 'clipiqa', 'musiq')
    ms = {k: pyiqa.create_metric(k, device=dev) for k in keys}
    print(f'[metric] {list(ms)} ready (device={dev})', flush=True)

    res = {k: {'sc04': [], 'm24': []} for k in keys}
    ts = time.time()
    for i, (n, p, q) in enumerate(pairs, 1):
        for tag, path in (('sc04', p), ('m24', q)):
            im = Image.open(path).convert('RGB')
            t = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            for k, m in ms.items():
                with torch.no_grad():
                    res[k][tag].append(float(m(t).item()))
        if i % 5 == 0:
            print(f'  {i}/{N} ({time.time()-ts:.0f}s)', flush=True)

    mean = {k: {t: round(float(np.mean(v)), 4) for t, v in d.items()} for k, d in res.items()}
    print(f'\n=== 测试集 {N} 张配对复核 (同 q95 jpg, 逐图同名对照) ===')
    print(f'  {"metric":<10}{"M24(sc0.8)":>12}{"sc0.4":>10}{"Δ":>10}')
    for k in keys:
        m0, m1 = mean[k]['m24'], mean[k]['sc04']
        print(f'  {k:<10}{m0:>12.4f}{m1:>10.4f}{m1-m0:>+10.4f}')

    nq0, nq1 = mean['niqe']['m24'], mean['niqe']['sc04']
    mu0, mu1 = mean['musiq']['m24'], mean['musiq']['sc04']
    p_niqe = 94.444 - 10.028 * nq1
    p_niqe0 = 94.444 - 10.028 * nq0
    p_2 = 81.498 + 0.245 * mu1 - 9.449 * nq1
    p_20 = 81.498 + 0.245 * mu0 - 9.449 * nq0
    print(f'\n  niqe-only 模型: M24={p_niqe0:.2f} -> sc0.4={p_niqe:.2f}  (Δ={p_niqe-p_niqe0:+.2f})')
    print(f'  2 参数模型   : M24={p_20:.2f} -> sc0.4={p_2:.2f}  (Δ={p_2-p_20:+.2f})')
    print(f'  （M24 官方实测 43.6951; 本复核为探针 A 提交前的最后一道证据门）')

    OUT.write_text(json.dumps({
        'n': N, 'names': names, 'mean': mean,
        'per_image': {k: {t: [round(x, 4) for x in v] for t, v in d.items()} for k, d in res.items()},
        'pred_niqe_only': {'m24': round(p_niqe0, 3), 'sc04': round(p_niqe, 3)},
        'pred_2param': {'m24': round(p_20, 3), 'sc04': round(p_2, 3)},
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n已存 {OUT}')
    print('VERIFY_PROBE_A_DONE')


if __name__ == '__main__':
    main()
