# -*- coding: utf-8 -*-
"""原生分辨率裁切对照: 测试集 LQ vs 5 个已交包的输出 (同位置 512x512 原生像素)"""
from pathlib import Path
from PIL import Image, ImageDraw

T = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
PKGS = [
    ('LQ(input)', None),
    ('M24', Path(r'D:/models/final_compliant/output_dir')),
    ('probeA', Path(r'D:/models/final_compliant_probeA_sc04/output_dir')),
    ('sc00', Path(r'D:/models/final_compliant_M24sc00/output_dir')),
    ('avg2', Path(r'D:/models/final_compliant_M24avg2/output_dir')),
    ('p4', Path(r'D:/models/final_compliant_M24p4/output_dir')),
]
CASES = [13, 2]          # 绿植 / 建筑
S = 512
x0 = 1500; y0 = 1200     # 原生坐标原点(各图相同位置)

rows = []
for c in CASES:
    lq = T / f'case{c}.jpg'
    im0 = Image.open(lq).convert('RGB')
    for _, d in PKGS[1:]:
        p = d / f'case{c}.jpg'
        if p.exists():
            im0 = min(im0, Image.open(p).convert('RGB'), key=lambda i: i.size[0]*i.size[1]) if False else im0
    for name, d in PKGS:
        p = T / f'case{c}.jpg' if d is None else d / f'case{c}.jpg'
        if not p.exists():
            rows.append((f'case{c} {name} MISSING', Image.new('RGB', (S, S), 'red'))); continue
        im = Image.open(p).convert('RGB')
        w, h = im.size
        cx = min(x0, w - S); cy = min(y0, h - S)
        rows.append((f'case{c} {name}', im.crop((cx, cy, cx+S, cy+S))))

cols = len(PKGS)
sheet = Image.new('RGB', (S*cols, (S+18)*len(CASES)), 'white')
dr = ImageDraw.Draw(sheet)
for i, (name, im) in enumerate(rows):
    r, cc = divmod(i, cols)
    sheet.paste(im, (cc*S, r*(S+18)))
    dr.text((cc*S+4, r*(S+18)+S+3), name, fill='black')
out = Path(r'D:/tmp_timing/native_crops.png')
sheet.save(out)
print('saved', out, sheet.size)
