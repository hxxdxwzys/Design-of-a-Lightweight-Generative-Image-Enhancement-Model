# -*- coding: utf-8 -*-
"""HYPIR 图在测试集上的质量代理 (clipiqa/niqe) -- 判断它是否比 M24 更好的质量资产"""
import os, json
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
DIRS = [('HYPIR', r'D:/models/hypir/out_test100/result'),
        ('M24', r'D:/models/final_compliant/output_dir')]
dev = 'cuda'
ms = {k: pyiqa.create_metric(k, device=dev) for k in ('niqe', 'clipiqa')}
names = [f.stem for f in sorted(TEST.glob('*.jpg'))[:30]]
out = {}
for pkg, d in DIRS:
    D = Path(d)
    if not D.exists(): continue
    v = {k: [] for k in ms}
    for n in names:
        p = D / f'{n}.jpg'
        if not p.exists(): p = D / f'{n}.png'
        if not p.exists(): continue
        t = torch.from_numpy(np.asarray(Image.open(p).convert('RGB'))).permute(2,0,1).unsqueeze(0).to(dev).float().div_(255)
        for k, m in ms.items():
            with torch.no_grad(): v[k].append(float(m(t).item()))
        del t
    out[pkg] = {k: round(float(np.mean(x)), 4) for k, x in v.items()}
    out[pkg]['n'] = len(v['niqe'])
    print(pkg, json.dumps(out[pkg]), flush=True)
print('ref: M24 clipiqa 0.9023/niqe 5.2559 | TinyV1 0.3367 | TinyV2 0.3532 | LQ 0.2752/7.8100', flush=True)
Path(r'D:/tmp_timing/hypir_quality.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print('HYPIR_Q_DONE')
