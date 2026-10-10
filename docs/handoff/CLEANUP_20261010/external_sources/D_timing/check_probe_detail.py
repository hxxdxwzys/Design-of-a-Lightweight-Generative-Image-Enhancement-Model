import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
D = Path(r'D:/models/probe_detail/output_dir')
IDX = list(range(1, 101, 5))
lv = []
for i in IDX:
    g = np.asarray(Image.open(D / f'case{i}.jpg').convert('L'), dtype=np.float32)
    lv.append(float(ndimage.laplace(g).var()))
print('probe_detail lapvar =', round(float(np.mean(lv)), 1), ' (M24 = 762.4, x%.2f)' % (np.mean(lv)/762.4))
print('files =', len(list(D.glob('*.jpg'))))
