# -*- coding: utf-8 -*-
"""补全 MANIQA（最后一个盲点指标）: 对 5 个提交包 + LQ + val GT/LQ 全量测
2026-09-11: 代理打通后 MANIQA 可用
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROXY = 'http://127.0.0.1:7890'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ[k] = PROXY

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
PKGS = {
    'M24': Path(r'D:/models/final_compliant/output_dir'),
    'm0s1': Path(r'D:/models/final_compliant_14b/output_dir'),
    'g3w18s1': Path(r'D:/models/final_compliant_g3w18s1/output_dir'),
    'm0s7_sc10': Path(r'D:/models/final_compliant_m0s7/output_dir'),
    'm0s7_s097': Path(r'D:/models/wave2_m0s7_s097/output_dir'),
}
OUT = Path(r'D:/tmp_timing/maniqa_all.json')
N = 100


def main():
    import numpy as np
    import torch
    import pyiqa
    from PIL import Image

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric = pyiqa.create_metric('maniqa', device=dev)
    print('[maniqa] ready', flush=True)
    res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}

    def run(name, files, n=N):
        files = list(files)[:n]
        if not files:
            print(f'[skip] {name}'); return
        vals = []
        t0 = time.time()
        for j in files:
            im = Image.open(j).convert('RGB')
            t = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                vals.append(float(metric(t).item()))
        res[name] = {'maniqa': round(float(np.mean(vals)), 4),
                     'std': round(float(np.std(vals)), 4), 'n': len(vals)}
        OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
        print(f'[{name}] n={len(vals)} MANIQA={res[name]["maniqa"]:.4f} '
              f'±{res[name]["std"]:.4f} ({time.time()-t0:.0f}s)', flush=True)

    for k, d in PKGS.items():
        if d.exists():
            run(f'{k}_test', sorted(d.glob('*.jpg')))
    run('LQ_test', sorted(TEST.glob('*.jpg')))
    run('GT_val', sorted(VAL.glob('*_gt.*')), n=3)
    run('LQ_val', sorted(VAL.glob('*_lq.*')), n=3)

    print('\n=== MANIQA 汇总（对照官方分）===')
    SCORE = {'M24': 43.684, 'm0s1': 34.102, 'g3w18s1': 31.819,
             'm0s7_sc10': 30.608, 'm0s7_s097': 31.165}
    for k in PKGS:
        v = res.get(f'{k}_test', {}).get('maniqa')
        s = SCORE.get(k)
        print(f'  {k:<11} MANIQA={v if v is None else round(v,4)}   官方分={s}')
    for k in ('LQ_test', 'GT_val', 'LQ_val'):
        if k in res:
            print(f'  {k:<11} MANIQA={res[k]["maniqa"]}')
    print(f'\n已存 {OUT}')


if __name__ == '__main__':
    main()
