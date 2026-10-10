# -*- coding: utf-8 -*-
"""clipiqa 确定性体检 (CPU, 不占显存): 同一张图重复测, 以及不同目录同内容图对比"""
import os, sys, time
os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image
DEV = 'cpu'
m = pyiqa.create_metric('clipiqa', device=DEV)

def score(path):
    im = Image.open(path).convert('RGB')
    t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).unsqueeze(0).to(DEV).float().div_(255)
    with torch.no_grad():
        return float(m(t).item())

tests = [
    ('traj/v7g25k case1 (第1次)', r'D:/models/traj/v7g25k/case1.jpg'),
    ('traj/v7g25k case1 (第2次)', r'D:/models/traj/v7g25k/case1.jpg'),
    ('traj/v7g25k case10(第1次)', r'D:/models/traj/v7g25k/case10.jpg'),
    ('probe_native case1', r'D:/models/probe_native/case1.jpg'),
]
for name, p in tests:
    t0 = time.time()
    print('%-28s clipiqa=%.4f  (%.1fs)' % (name, score(p), time.time() - t0), flush=True)
print('CLIPIQA_CHECK_DONE')
