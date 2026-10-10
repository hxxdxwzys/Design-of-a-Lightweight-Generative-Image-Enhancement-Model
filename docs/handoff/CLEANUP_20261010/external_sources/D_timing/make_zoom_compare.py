# -*- coding: utf-8 -*-
"""2x 放大裁切对照: LQ / M24 / probeA / sc00 / avg2 (256 原生 -> 512 显示)"""
from pathlib import Path
from PIL import Image, ImageDraw
T = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
COLS = [('LQ', None), ('M24', r'D:/models/final_compliant/output_dir'),
        ('probeA', r'D:/models/final_compliant_probeA_sc04/output_dir'),
        ('sc00', r'D:/models/final_compliant_M24sc00/output_dir'),
        ('avg2', r'D:/models/final_compliant_M24avg2/output_dir')]
CASES = [(13, (1500, 1200)), (2, (1800, 2000)), (50, (900, 600))]
S = 256
rows = []
for c, (x, y) in CASES:
    for n, d in COLS:
        p = (T / f'case{c}.jpg') if d is None else (Path(d) / f'case{c}.jpg')
        if not p.exists():
            rows.append((f'c{c} {n} MISS', Image.new('RGB', (S, S), 'red'))); continue
        im = Image.open(p).convert('RGB')
        w, h = im.size
        cx, cy = min(x, w-S), min(y, h-S)
        rows.append((f'case{c} {n}', im.crop((cx, cy, cx+S, cy+S)).resize((S*2, S*2), Image.NEAREST)))
sheet = Image.new('RGB', (S*2*len(COLS), (S*2+18)*len(CASES)), 'white')
dr = ImageDraw.Draw(sheet)
for i, (nm, im) in enumerate(rows):
    r, cc = divmod(i, len(COLS))
    sheet.paste(im, (cc*S*2, r*(S*2+18)))
    dr.text((cc*S*2+4, r*(S*2+18)+S*2+3), nm, fill='black')
out = Path(r'D:/tmp_timing/zoom_compare.png'); sheet.save(out); print(out, sheet.size)
