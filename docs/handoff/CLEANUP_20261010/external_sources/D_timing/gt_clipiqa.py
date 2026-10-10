# -*- coding: utf-8 -*-
"""关键校准: val GT / val LQ / M24输出 / PixRestore输出 的 clipiqa (同口径 512 中心裁切)"""
import os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image
VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
m = pyiqa.create_metric('clipiqa', device='cuda')
def cq(p, crop=None):
    im = Image.open(p).convert('RGB')
    if crop:
        W, H = im.size
        box = ((W-512)//2, (H-512)//2, (W-512)//2+512, (H-512)//2+512)
        im = im.crop(box)
    t = torch.from_numpy(np.asarray(im)).permute(2,0,1).unsqueeze(0).cuda().float().div_(255)
    with torch.no_grad():
        return float(m(t).item())
rows = {'val_GT': [], 'val_LQ': [], 'M24_on_val': [], 'PixB_on_val': []}
for c in (1,2,3):
    rows['val_GT'].append(cq(next(VAL.glob(f'case{c}_gt.*')), crop=True))
    rows['val_LQ'].append(cq(next(VAL.glob(f'case{c}_lq.*')), crop=True))
    p = Path(r'D:/tmp_timing/val_over_sharp') / f'case{c}_lq.png'
    if p.exists(): rows['M24_on_val'].append(cq(p, crop=True))
for k, v in rows.items():
    if v: print(f'{k:14s} clipiqa = {np.mean(v):.4f}  (逐张 {[round(x,3) for x in v]})', flush=True)
print()
print('参照(测试集): M24 0.9023 | PixRestore-b 0.3055 | TinyV2 0.3532 | LQ 0.2752', flush=True)
print('GT_CLIPIQA_DONE')
