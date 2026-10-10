# -*- coding: utf-8 -*-
"""诊断: 微模型在测试集上的无参指标 vs M24 (官方分差 14-16 分, 需找出原因)"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
import numpy as np
import torch
import pyiqa
from PIL import Image

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
DIRS = [
    ('M24',     r'D:/models/final_compliant/output_dir'),
    ('TinyV1',  r'D:/models/pkg_TinyV1/output_dir'),
    ('TinyV2',  r'D:/models/pkg_TinyV2/output_dir'),
    ('TinyV4',  r'D:/models/pkg_TinyV4/output_dir'),
    ('LQ',      str(TEST)),
]
KEYS = ('niqe', 'clipiqa')
OUT = Path(r'D:/tmp_timing/tiny_quality_diag.jsonl')
dev = 'cuda'
ms = {k: pyiqa.create_metric(k, device=dev) for k in KEYS}
names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:30]]
print(f'[start] {len(names)} 张 x {len(DIRS)} 组 x {len(KEYS)} 指标', flush=True)
agg = {}
with OUT.open('a', encoding='utf-8') as fo:
    for pkg, d in DIRS:
        vals = {k: [] for k in KEYS}
        rgb_mean, rgb_std, lv = [], [], []
        for n in names:
            p = Path(d) / f'{n}.jpg'
            if not p.exists():
                continue
            im = Image.open(p).convert('RGB')
            arr = np.asarray(im)
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
            rec = {'pkg': pkg, 'name': n}
            for k, m in ms.items():
                with torch.no_grad():
                    rec[k] = round(float(m(t).item()), 5)
                vals[k].append(rec[k])
            rgb_mean.append(arr.reshape(-1, 3).mean(0)); rgb_std.append(arr.reshape(-1, 3).std(0))
            g = np.asarray(im.convert('L'), dtype=np.float32)
            from scipy import ndimage
            lv.append(float(ndimage.laplace(g).var()))
            fo.write(json.dumps(rec) + '\n'); fo.flush()
            del im, arr, t
        agg[pkg] = {k: float(np.mean(v)) for k, v in vals.items()}
        agg[pkg]['rgb_mean'] = [round(float(x), 2) for x in np.mean(rgb_mean, 0)]
        agg[pkg]['rgb_std'] = [round(float(x), 2) for x in np.mean(rgb_std, 0)]
        agg[pkg]['lapvar'] = round(float(np.mean(lv)), 1)
        print(pkg, agg[pkg], flush=True)
Path(r'D:/tmp_timing/tiny_quality_diag.json').write_text(json.dumps(agg, indent=1, ensure_ascii=False), encoding='utf-8')
print('DIAG_DONE')
