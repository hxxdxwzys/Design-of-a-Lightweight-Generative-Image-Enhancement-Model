# -*- coding: utf-8 -*-
"""A(提前版): HYPIR 已完成的图 vs M24 同图 —— 配对比较细节能量
HYPIR 输出为 PNG(无损), M24 为 q98 jpg; 另把 HYPIR 转存 q98 jpg 做公平对照。
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import io
from scipy import ndimage

HYP = Path(r'D:/models/hypir/out_test100/result')
M24 = Path(r'D:/models/final_compliant/output_dir')
LQ = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')

def feats(im):
    g = np.asarray(im.convert('L'), dtype=np.float32)
    return float(ndimage.laplace(g).var()), float(np.sqrt(np.mean((g - ndimage.gaussian_filter(g, 2.0)) ** 2)))

done = sorted(HYP.glob('*.png'))
print('HYPIR done:', len(done))
res = []
for p in done:
    stem = p.stem
    m = M24 / f'{stem}.jpg'
    l = LQ / f'{stem}.jpg'
    if not m.exists() or not l.exists():
        continue
    hi = Image.open(p).convert('RGB')
    b = io.BytesIO(); hi.save(b, format='JPEG', quality=98, subsampling=0); b.seek(0)
    hv, hf = feats(Image.open(b))
    mv, mf = feats(Image.open(m))
    lv, lf = feats(Image.open(l))
    res.append((stem, lv, mv, hv, lf, mf, hf))
    if len(res) <= 5 or len(res) % 10 == 0:
        print(f'{stem:8s} LQ {lv:7.1f} | M24 {mv:7.1f} | HYPIR {hv:7.1f}  (HYPIR/M24={hv/max(mv,1e-6):.2f})', flush=True)

if res:
    a = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in res], dtype=float)
    print(f'\n配对 n={len(res)}')
    print(f'  LQ    lapvar={a[:,0].mean():8.1f}  hf={a[:,3].mean():6.3f}')
    print(f'  M24   lapvar={a[:,1].mean():8.1f}  hf={a[:,4].mean():6.3f}')
    print(f'  HYPIR lapvar={a[:,2].mean():8.1f}  hf={a[:,5].mean():6.3f}')
    print(f'  HYPIR/M24 lapvar 比值中位 = {np.median(a[:,2]/a[:,1]):.2f}')
    print(f'  HYPIR 更锐的图: {(a[:,2]>a[:,1]).sum()}/{len(res)}')
    Path(r'D:/tmp_timing/hypir_vs_m24_detail.json').write_text(json.dumps(
        [{'case': r[0], 'lq': r[1], 'm24': r[2], 'hypir': r[3]} for r in res], indent=1), encoding='utf-8')
print('DONE')
