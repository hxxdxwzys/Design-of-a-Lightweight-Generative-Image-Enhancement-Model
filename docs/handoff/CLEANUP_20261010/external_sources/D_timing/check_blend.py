import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
IDX = list(range(1, 101, 5))
def lv(d):
    a=[]
    for i in IDX:
        p = Path(d)/f'case{i}.jpg'
        if p.exists():
            g = np.asarray(Image.open(p).convert('L'), dtype=np.float32)
            a.append(float(ndimage.laplace(g).var()))
    return float(np.mean(a))
for name, d in [('M24 base', r'D:/models/final_compliant/output_dir'),
                ('ProbeBlend(0.85)', r'D:/models/probe_blend/output_dir'),
                ('ProbeDetail(x2.27)', r'D:/models/probe_detail/output_dir')]:
    v = lv(d); print(f'{name:22s} lapvar={v:8.1f}')
