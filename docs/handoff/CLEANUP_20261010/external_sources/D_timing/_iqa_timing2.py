# -*- coding: utf-8 -*-
"""给 IQA 指标在 4K 上的单图耗时做正确计时 (先建好模型, 再计时)."""
import sys, time
from pathlib import Path
import numpy as np, torch, pyiqa
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
ms = {k: pyiqa.create_metric(k, device='cuda') for k in ('clipiqa','musiq','maniqa','niqe')}
files = sorted(TEST.glob('*.jpg'))[:4]
ts = {k: [] for k in ms}
for f in files:
    t = torch.from_numpy(np.asarray(Image.open(f).convert('RGB'))).permute(2,0,1).unsqueeze(0).cuda().float().div_(255)
    for k, m in ms.items():
        t0 = time.time()
        with torch.no_grad(): v = float(m(t).item())
        ts[k].append((time.time()-t0, v))
    del t
for k, v in ts.items():
    print('%-8s 单图: %s  | 值 %s' % (k, ' '.join('%.2fs' % x[0] for x in v), ' '.join('%.4f' % x[1] for x in v)))
print('TIMING2_DONE')
