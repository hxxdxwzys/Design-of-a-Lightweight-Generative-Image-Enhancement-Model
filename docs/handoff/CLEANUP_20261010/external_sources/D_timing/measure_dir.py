# -*- coding: utf-8 -*-
"""通用目录质量代理 (2026-09-15): 对任意目录的测试集出图算 niqe/clipiqa/lapvar
用法: python measure_dir.py --dir DIR --pattern "case{n}.png" --n 30 --metrics niqe,clipiqa,lapvar
"""
import argparse, json, os, sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
for _k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(_k, None)
import numpy as np
import torch
import pyiqa
from PIL import Image

TEST = Path('C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True)
ap.add_argument('--pattern', default='case{n}.jpg')
ap.add_argument('--n', type=int, default=30)
ap.add_argument('--metrics', default='niqe,clipiqa,lapvar')
ap.add_argument('--tag', default='')
a = ap.parse_args()
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
dev = 'cuda'
want = [m.strip() for m in a.metrics.split(',') if m.strip()]
iqa = {k: pyiqa.create_metric(k, device=dev) for k in want if k != 'lapvar'}
names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:a.n]]
vals = {k: [] for k in want}
for nm in names:
    idx = ''.join(ch for ch in nm if ch.isdigit())
    p = Path(a.dir) / a.pattern.format(n=idx)
    if not p.exists():
        continue
    im = Image.open(p).convert('RGB')
    t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
    for k, m in iqa.items():
        with torch.no_grad():
            vals[k].append(float(m(t).item()))
    if 'lapvar' in vals:
        g = np.asarray(im.convert('L'), dtype=np.float64)
        lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
        vals['lapvar'].append(float(lap.var()))
    del t
out = {k: round(float(np.mean(v)), 4) for k, v in vals.items() if v}
out['n'] = len(next(iter([v for v in vals.values() if v])))
out['dir'] = a.dir
print('[measure]', json.dumps(out, ensure_ascii=False), flush=True)
Path(Path(a.dir).parent / ('metrics_%s.json' % (a.tag or Path(a.dir).parent.name))).write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('MEASURE_DIR_DONE')
