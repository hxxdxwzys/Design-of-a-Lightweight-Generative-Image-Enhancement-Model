# -*- coding: utf-8 -*-
"""量化 q98 vs q95 对 lapvar 的影响 (消除 M24 的编码混淆)"""
from pathlib import Path
import io
import numpy as np
from PIL import Image
from scipy import ndimage

def lapvar(im):
    g = np.asarray(im.convert('L'), dtype=np.float32)
    return float(ndimage.laplace(g).var())

IDX = list(range(1, 101, 5))
src = Path(r'D:/models/final_compliant/output_dir')
a98, a95 = [], []
for i in IDX:
    im = Image.open(src / f'case{i}.jpg').convert('RGB')
    a98.append(lapvar(im))
    b = io.BytesIO(); im.save(b, format='JPEG', quality=95, subsampling=2); b.seek(0)
    a95.append(lapvar(Image.open(b)))
a98 = np.array(a98); a95 = np.array(a95)
print(f'M24  保存 q98: lapvar={a98.mean():.1f}')
print(f'M24  转存 q95: lapvar={a95.mean():.1f}   ({(a95.mean()/a98.mean()-1)*100:+.1f}%)')
print(f'-> 与 q95 组的 probeA(876.1) 对比: q95 口径下 M24={a95.mean():.1f} vs probeA=876.1')
