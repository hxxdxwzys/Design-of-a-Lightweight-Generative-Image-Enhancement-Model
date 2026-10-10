# -*- coding: utf-8 -*-
"""Val 验证集: LQ/G
T 的细节能量 (参照系) —— 我们输出的 x10.8 到底是欠锐还是过锐?"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')

def feats(p):
    g = np.asarray(Image.open(p).convert('L'), dtype=np.float32)
    lv = float(ndimage.laplace(g).var())
    gy, gx = np.gradient(g)
    gm = float(np.mean(np.hypot(gx, gy)))
    hf = float(np.sqrt(np.mean((g - ndimage.gaussian_filter(g, 2.0)) ** 2)))
    return lv, gm, hf

out = {}
for tag, pat in (('val_LQ', '*_lq.*'), ('val_GT', '*_gt.*')):
    vals = []
    for p in sorted(VAL.glob(pat)):
        f = feats(p); vals.append(f)
        print(f'  {tag} {p.name}: lapvar={f[0]:.1f} grad={f[1]:.3f} hf={f[2]:.3f}', flush=True)
    a = np.array(vals); out[tag] = {'lapvar': float(a[:,0].mean()), 'grad': float(a[:,1].mean()), 'hf': float(a[:,2].mean())}
print('\nval 均值:', json.dumps(out, ensure_ascii=False, indent=1))

# 测试集 LQ (与包输出同 20 张, 做同图对照)
idx = list(range(1, 101, 5))
vals = []
for i in idx:
    p = TEST / f'case{i}.jpg'
    if p.exists(): vals.append(feats(p))
a = np.array(vals)
out['test_LQ_20'] = {'lapvar': float(a[:,0].mean()), 'grad': float(a[:,1].mean()), 'hf': float(a[:,2].mean())}
print('test_LQ_20:', out['test_LQ_20'])

# val 上各包的同图增益需要渲染, 这里先给参照
for tag in ('val_LQ','val_GT'):
    v = out[tag]
    print(f'{tag}: lapvar={v["lapvar"]:.1f} grad={v["grad"]:.3f} hf={v["hf"]:.3f}')
Path(r'D:/tmp_timing/val_detail_ref.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print('DONE')
