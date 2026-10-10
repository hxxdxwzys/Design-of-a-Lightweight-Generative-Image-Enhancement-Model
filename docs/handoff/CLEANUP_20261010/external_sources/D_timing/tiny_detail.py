import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
IDX = list(range(1, 101, 5))
def stats(d):
    lv, hf = [], []
    for i in IDX:
        g = np.asarray(Image.open(Path(d)/f'case{i}.jpg').convert('L'), dtype=np.float32)
        lv.append(float(ndimage.laplace(g).var()))
        hf.append(float(np.sqrt(np.mean((g - ndimage.gaussian_filter(g, 2.0))**2))))
    return float(np.mean(lv)), float(np.mean(hf))
for name, d in [('LQ 输入', r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'),
                ('M24 (0.78x)', r'D:/models/final_compliant/output_dir'),
                ('ProbeBlend (40.54)', r'D:/models/pkg_TinyV1/../pkg_TinyV2/output_dir'),
                ('TinyV1 (6.1x)', r'D:/models/pkg_TinyV1/output_dir'),
                ('TinyV2 (19.5x)', r'D:/models/pkg_TinyV2/output_dir')]:
    if 'ProbeBlend' in name: continue
    lv, hf = stats(d)
    print(f'{name:22s} lapvar={lv:8.1f}  hf={hf:6.3f}')
