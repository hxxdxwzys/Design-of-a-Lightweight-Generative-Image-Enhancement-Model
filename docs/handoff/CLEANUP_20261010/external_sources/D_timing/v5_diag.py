import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
IDX = list(range(1, 101, 5))
def stats(d):
    lv, hf, mn, sd = [], [], [], []
    for i in IDX:
        p = Path(d)/f'case{i}.jpg'
        if not p.exists(): p = Path(d)/f'case{i}.png'
        if not p.exists(): continue
        im = Image.open(p).convert('RGB')
        g = np.asarray(im.convert('L'), dtype=np.float32)
        lv.append(float(ndimage.laplace(g).var()))
        hf.append(float(np.sqrt(np.mean((g - ndimage.gaussian_filter(g, 2.0))**2))))
        a = np.asarray(im, dtype=np.float32).reshape(-1,3)
        mn.append(a.mean(0)); sd.append(a.std(0))
    return np.mean(lv), np.mean(hf), np.mean(mn,0), np.mean(sd,0)
for n, d in [('LQ', r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'),
             ('M24', r'D:/models/final_compliant/output_dir'),
             ('TinyV2 (0.353)', r'D:/models/pkg_TinyV2/output_dir'),
             ('TinyV5d (0.390)', r'D:/models/pkg_TinyV5d/output_dir')]:
    lv, hf, mn, sd = stats(d)
    print(f'{n:18s} lapvar={lv:7.1f} hf={hf:6.2f}  mean={mn.round(1)} std={sd.round(1)}')
