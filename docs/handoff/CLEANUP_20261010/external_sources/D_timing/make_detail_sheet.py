# -*- coding: utf-8 -*-
"""并排对照: val LQ / val GT / 测试集 LQ  (各取中心 1024 区域缩到 384)"""
from pathlib import Path
from PIL import Image, ImageDraw

V = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
T = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')

def crop384(p):
    im = Image.open(p).convert('RGB')
    w, h = im.size
    s = 1536
    x = (w - s)//2; y = (h - s)//2
    return im.crop((x, y, x+s, y+s)).resize((384, 384), Image.LANCZOS)

rows = [
    ('val LQ c1', V/'case1_lq.jpg'), ('val GT c1', V/'case1_gt.png'),
    ('test LQ case1', T/'case1.jpg'), ('test LQ case10', T/'case10.jpg'),
    ('val LQ c2', V/'case2_lq.jpg'), ('val GT c2', V/'case2_gt.png'),
    ('test LQ case11', T/'case11.jpg'), ('test LQ case13', T/'case13.jpg'),
]
sheet = Image.new('RGB', (384*4, 384*2+40), 'white')
d = ImageDraw.Draw(sheet)
for i, (name, p) in enumerate(rows):
    r, c = divmod(i, 4)
    sheet.paste(crop384(p), (c*384, r*(384+20)))
    d.text((c*384+4, r*(384+20)+384+2), name, fill='black')
out = Path(r'D:/tmp_timing/detail_sheet.png')
sheet.save(out)
print('saved', out, sheet.size)
