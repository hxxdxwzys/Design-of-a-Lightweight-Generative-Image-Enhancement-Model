from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
T = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
ROWS = [('LQ', None), ('M24', r'D:/models/final_compliant/output_dir'),
        ('TinyV2', r'D:/models/pkg_TinyV2/output_dir')]
CASES = [1, 20, 60, 99]
W = 320
tiles = []
for c in CASES:
    for n, d in ROWS:
        p = (T / f'case{c}.jpg') if d is None else (Path(d) / f'case{c}.jpg')
        im = Image.open(p).convert('RGB')
        im.thumbnail((W, W), Image.LANCZOS)
        tiles.append((f'case{c} {n}', im))
H = max(t[1].height for t in tiles)
sheet = Image.new('RGB', (W*len(ROWS), (H+16)*len(CASES)), 'white')
dr = ImageDraw.Draw(sheet)
for i, (nm, im) in enumerate(tiles):
    r, cc = divmod(i, len(ROWS))
    sheet.paste(im, (cc*W, r*(H+16)))
    dr.text((cc*W+4, r*(H+16)+H+2), nm, fill='black')
out = Path(r'D:/tmp_timing/full_compare.png'); sheet.save(out); print(out, sheet.size)
# 顺带打印全局色彩统计
for c in CASES:
    line = []
    for n, d in ROWS:
        p = (T / f'case{c}.jpg') if d is None else (Path(d) / f'case{c}.jpg')
        a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32).reshape(-1, 3)
        line.append(f'{n}: mean={a.mean(0).round(1)} std={a.std(0).round(1)}')
    print(f'case{c}  ' + ' | '.join(line))
