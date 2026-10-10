import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
PRED = Path(r'D:/tmp_timing/val_over_sharp')

def feats(p):
    im = Image.open(p).convert('L')
    g = np.asarray(im, dtype=np.float32)
    lap = float(ndimage.laplace(g).var())
    hf = float(np.sqrt(np.mean((g - ndimage.gaussian_filter(g, 2.0)) ** 2)))
    return lap, hf, im.size

rows = []
for c in (1, 2, 3):
    lq = next(VAL.glob(f'case{c}_lq.*'))
    gt = next(VAL.glob(f'case{c}_gt.*'))
    pr = PRED / f'case{c}_lq.png'
    a, b, d = feats(lq), feats(gt), feats(pr)
    rows.append((a, b, d))
    print(f'case{c}: LQ lap={a[0]:7.1f} hf={a[1]:5.2f} | GT lap={b[0]:7.1f} hf={b[1]:5.2f} | '
          f'OUR lap={d[0]:8.1f} hf={d[1]:5.2f}  OUR/GT lap={d[0]/b[0]:5.2f} hf={d[1]/b[1]:4.2f} | sizes {a[2]} {b[2]} {d[2]}')
a = np.array([[r[0][0], r[0][1]] for r in rows]); b = np.array([[r[1][0], r[1][1]] for r in rows]); d = np.array([[r[2][0], r[2][1]] for r in rows])
print(f'\n均值 lapvar: LQ={a[:,0].mean():.1f}  GT={b[:,0].mean():.1f}  OUR={d[:,0].mean():.1f}  (OUR/GT={d[:,0].mean()/b[:,0].mean():.2f})')
print(f'均值 hf    : LQ={a[:,1].mean():.2f}  GT={b[:,1].mean():.2f}  OUR={d[:,1].mean():.2f}  (OUR/GT={d[:,1].mean()/b[:,1].mean():.2f})')
