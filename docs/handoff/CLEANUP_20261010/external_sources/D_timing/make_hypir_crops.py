from pathlib import Path
from PIL import Image, ImageDraw
H = Path(r'D:/models/hypir/out_test100/result')
M = Path(r'D:/models/final_compliant/output_dir')
L = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
CASES = [35, 12, 100]
X, Y, S = 1200, 900, 512
sheet = Image.new('RGB', (S*3, (S+16)*len(CASES)), 'white')
d = ImageDraw.Draw(sheet)
for r, c in enumerate(CASES):
    pairs = [('LQ', L/f'case{c}.jpg'), ('M24', M/f'case{c}.jpg'), ('HYPIR', H/f'case{c}.png')]
    for ci, (n, p) in enumerate(pairs):
        im = Image.open(p).convert('RGB')
        w, h = im.size
        x = min(X, w-S); y = min(Y, h-S)
        sheet.paste(im.crop((x, y, x+S, y+S)), (ci*S, r*(S+16)))
        d.text((ci*S+4, r*(S+16)+S+2), f'case{c} {n} {im.size}', fill='black')
out = Path(r'D:/tmp_timing/hypir_vs_m24_crops.png'); sheet.save(out); print(out, sheet.size)
