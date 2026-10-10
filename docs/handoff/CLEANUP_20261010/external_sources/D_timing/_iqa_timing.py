# -*- coding: utf-8 -*-
import os, sys, time
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import numpy as np, torch, pyiqa
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
p = sorted(TEST.glob('*.jpg'))[0]
im = Image.open(p).convert('RGB')
print('size', im.size, flush=True)
t0=time.time(); arr = np.asarray(im); print('load %.2fs' % (time.time()-t0), flush=True)
t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).cuda().float().div_(255)
for k in ('clipiqa','musiq','maniqa','niqe'):
    t0=time.time()
    try:
        m = pyiqa.create_metric(k, device='cuda')
        v = float(m(t).item())
        print('%-8s %.3fs  val=%.4f' % (k, time.time()-t0, v), flush=True)
    except Exception as e:
        print('%-8s FAIL %r' % (k, repr(e)[:150]), flush=True)
# 512 resize timing
t0=time.time()
ts = torch.nn.functional.interpolate(t, size=(512,512), mode='bicubic', align_corners=False)
print('resize512 %.2fs' % (time.time()-t0), flush=True)
print('TIMING_DONE')
