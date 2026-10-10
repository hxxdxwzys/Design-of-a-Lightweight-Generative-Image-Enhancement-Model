# -*- coding: utf-8 -*-
"""计算新模型 (cnn_out_f2k) 100 张测试图 4K clipiqa 均值, 对比 LQ 与 s6k (取自 per_image_report.json)"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pyiqa

rep = json.load(open(r'D:/tmp_timing/per_image_report.json'))
tc = rep['test_clipiqa']
lq_mean = float(np.mean([v['lq'] for v in tc.values()]))
s6k_mean = float(np.mean([v['s6k'] for v in tc.values()]))
print(f'[ref] LQ test clipiqa mean = {lq_mean:.4f}', flush=True)
print(f'[ref] s6k test clipiqa mean = {s6k_mean:.4f}', flush=True)

device = 'cuda'
cq = pyiqa.create_metric('clipiqa', device=device)
out = Path(r'D:/models/cnn_out_f2k')
vals = {}
for p in sorted(out.glob('*.jpg')):
    a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).clamp(1e-4, 1 - 1e-4).to(device)
    vals[p.stem] = float(cq(x).mean().item())

f2k_mean = float(np.mean(list(vals.values())))
print(f'[f2k] test clipiqa mean = {f2k_mean:.4f}', flush=True)
print(f'[delta] vs LQ: {f2k_mean - lq_mean:+.4f}', flush=True)
print(f'[delta] vs s6k: {f2k_mean - s6k_mean:+.4f}', flush=True)
json.dump({'lq_mean': lq_mean, 's6k_mean': s6k_mean, 'f2k_mean': f2k_mean, 'per_image': vals},
          open(r'D:/tmp_timing/test_clipiqa_f2k.json', 'w'), indent=2)
print('[saved] test_clipiqa_f2k.json', flush=True)
