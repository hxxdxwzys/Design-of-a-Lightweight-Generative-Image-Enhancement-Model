# -*- coding: utf-8 -*-
"""探针 A 预验证指标 (2026-09-12): PNG -> q95 jpg (与提交管线同规格) -> niqe/clipiqa/musiq
与 M24(sc0.8) 在同 30 张上的值对比 (nr_test100_results.json 口径: sorted()[:30])
"""
from __future__ import annotations

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

import numpy as np
import torch
import pyiqa
from PIL import Image

PNG_DIR = Path(r'D:/models/probe_a/pred_sc0.4')
JPG_DIR = Path(r'D:/models/probe_a/pred_sc0.4_jpg95')
OUT = Path(r'D:/tmp_timing/probe_a_screen.json')
BASE30 = {'niqe': 5.2559, 'clipiqa': 0.9023, 'musiq': 46.1257}   # M24 sc0.8, 同 30 张

JPG_DIR.mkdir(parents=True, exist_ok=True)
n = 0
for p in sorted(PNG_DIR.glob('*.png')):
    d = JPG_DIR / f'{p.stem}.jpg'
    if not d.exists():
        Image.open(p).convert('RGB').save(d, quality=95)
        n += 1
print(f'[转换] {n} 张 PNG -> q95 jpg (总 {len(list(JPG_DIR.glob("*.jpg")))} 张)', flush=True)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
metrics = {}
for k in ('niqe', 'clipiqa', 'musiq'):
    metrics[k] = pyiqa.create_metric(k, device=dev)
    print(f'  [metric] {k} ready', flush=True)

vals = {k: [] for k in metrics}
t0 = time.time()
for j in sorted(JPG_DIR.glob('*.jpg')):
    im = Image.open(j).convert('RGB')
    t = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
    for k, m in metrics.items():
        with torch.no_grad():
            vals[k].append(float(m(t).item()))
    if len(vals['niqe']) % 10 == 0:
        print(f'  {len(vals["niqe"])} 张 ({time.time()-t0:.0f}s)', flush=True)

res = {k: round(float(np.mean(v)), 4) for k, v in vals.items()}
print('\n=== 测试集 30 张直证对比（同为 q95 jpg）===')
print(f'  {"":<14}{"niqe":>9}{"clipiqa":>10}{"musiq":>9}')
print(f'  {"M24 (sc0.8)":<14}{BASE30["niqe"]:>9.4f}{BASE30["clipiqa"]:>10.4f}{BASE30["musiq"]:>9.4f}')
print(f'  {"sc0.4":<14}{res["niqe"]:>9.4f}{res["clipiqa"]:>10.4f}{res["musiq"]:>9.4f}')
print(f'  {"Δ":<14}{res["niqe"]-BASE30["niqe"]:>+9.4f}{res["clipiqa"]-BASE30["clipiqa"]:>+10.4f}'
      f'{res["musiq"]-BASE30["musiq"]:>+9.4f}')
p2 = 81.498 + 0.245 * res['musiq'] - 9.449 * res['niqe']
p2b = 81.498 + 0.245 * BASE30['musiq'] - 9.449 * BASE30['niqe']
p1 = 94.444 - 10.028 * res['niqe']
p1b = 94.444 - 10.028 * BASE30['niqe']
print(f'\n  niqe-only 模型: M24={p1b:.2f} -> sc0.4={p1:.2f}  (Δ={p1-p1b:+.2f})')
print(f'  2 参数模型   : M24={p2b:.2f} -> sc0.4={p2:.2f}  (Δ={p2-p2b:+.2f})')
print(f'  （M24 官方实测 43.6951）')
OUT.write_text(json.dumps({'scale': 0.4, 'n': len(vals['niqe']), 'metrics': res,
                           'baseline_M24_30': BASE30,
                           'pred_niqe_only': round(p1, 3), 'pred_2param': round(p2, 3)},
                          indent=1), encoding='utf-8')
print(f'\n已存 {OUT}')
