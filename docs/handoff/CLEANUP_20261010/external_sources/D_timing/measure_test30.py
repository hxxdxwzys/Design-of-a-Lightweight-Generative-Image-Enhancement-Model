# -*- coding: utf-8 -*-
"""本地质量代理: 测试集 30 张 clipiqa/niqe (与官方质量项最相关的两个)"""
import argparse, json, os, sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--n', type=int, default=30)
a = ap.parse_args()
dev = 'cuda'
ms = {k: pyiqa.create_metric(k, device=dev) for k in ('niqe', 'clipiqa')}
names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:a.n]]
vals = {k: [] for k in ms}
for n in names:
    p = Path(a.dir) / f'{n}.jpg'
    if not p.exists(): continue
    t = torch.from_numpy(np.asarray(Image.open(p).convert('RGB'))).permute(2,0,1).unsqueeze(0).to(dev).float().div_(255)
    for k, m in ms.items():
        with torch.no_grad(): vals[k].append(float(m(t).item()))
    del t
out = {k: round(float(np.mean(v)), 4) for k, v in vals.items()}
out['n'] = len(vals['niqe'])
print('[measure]', json.dumps(out), flush=True)
print('  (参照: M24 clipiqa 0.9023 / niqe 5.2559 | TinyV1 0.3367 / 5.7899 | LQ 0.2752 / 7.8100)', flush=True)
Path(Path(a.dir).parent / 'test30_metrics.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print('MEASURE_TEST30_DONE')
