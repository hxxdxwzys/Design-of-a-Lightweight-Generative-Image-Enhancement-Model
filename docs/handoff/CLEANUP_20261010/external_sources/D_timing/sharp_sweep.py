# -*- coding: utf-8 -*-
"""锐化剂量 vs clipiqa: 平的微模型输出 + unsharp -> clipiqa 能回到 M24 档(0.90)吗?
只看本地代理, 不花提交次数.
"""
import os, json
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image, ImageFilter
from scipy import ndimage

SRC = Path(r'D:/models/pkg_TinyV5d/output_dir')       # v5 的 100 张 4K
OUT = Path(r'D:/tmp_timing/sharp_sweep'); OUT.mkdir(exist_ok=True)
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
DOSES = [0.0, 0.6, 1.2, 2.0]
IDX = [f'case{i}' for i in list(range(1, 101, 4))[:25]]   # 25 张
dev = 'cuda'
ms = {k: pyiqa.create_metric(k, device=dev) for k in ('clipiqa', 'niqe')}
print('dose  clipiqa   niqe   lapvar', flush=True)
for amt in DOSES:
    v = {k: [] for k in ms}; lv = []
    for n in IDX:
        p = SRC / f'{n}.jpg'
        if not p.exists(): continue
        im = Image.open(p).convert('RGB')
        if amt > 0:
            im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(amt*100), threshold=0))
        t = torch.from_numpy(np.asarray(im)).permute(2,0,1).unsqueeze(0).to(dev).float().div_(255)
        for k, m in ms.items():
            with torch.no_grad(): v[k].append(float(m(t).item()))
        g = np.asarray(im.convert('L'), dtype=np.float32)
        lv.append(float(ndimage.laplace(g).var()))
        del t
    row = {k: round(float(np.mean(x)), 4) for k, x in v.items()}
    row['lapvar'] = round(float(np.mean(lv)), 1); row['dose'] = amt
    print(f"{amt:4.1f}  {row['clipiqa']:.4f}  {row['niqe']:.4f}  {row['lapvar']:7.1f}", flush=True)
print('参照: M24 clipiqa 0.9023 lapvar 602.5 | v5 原始 0.3897 lapvar 126.9 | LQ 0.2752', flush=True)
print('SHARP_SWEEP_DONE')
