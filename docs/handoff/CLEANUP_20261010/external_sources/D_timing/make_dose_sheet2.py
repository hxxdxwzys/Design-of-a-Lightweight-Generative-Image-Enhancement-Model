from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
SRC = Path(r'D:/models/final_compliant/output_dir')
OUT = Path(r'D:/tmp_timing/dose_sweep/native'); OUT.mkdir(parents=True, exist_ok=True)
V = {'base': lambda im: im,
     'us_r1_p60': lambda im: im.filter(ImageFilter.UnsharpMask(1.0, 60, 0)),
     'us_r2_p120': lambda im: im.filter(ImageFilter.UnsharpMask(2.0, 120, 0)),
     'us_r2_p300': lambda im: im.filter(ImageFilter.UnsharpMask(2.0, 300, 0))}
X, Y, S = 1500, 1200, 512
for c in (13, 2):
    for n, fn in V.items():
        im = Image.open(SRC / f'case{c}.jpg').convert('RGB')
        fn(im).crop((X, Y, X+S, Y+S)).save(OUT / f'{n}_case{c}.png')
        fn(im).crop((X+1800, Y+1200, X+1800+S, Y+1200+S)).save(OUT / f'{n}_case{c}_b.png')
names = list(V)
sheet = Image.new('RGB', (S*len(names), (S+16)*4), 'white')
d = ImageDraw.Draw(sheet)
row = 0
for c in (13, 2):
    for suf in ('', '_b'):
        for ci, n in enumerate(names):
            p = OUT / f'{n}_case{c}{suf}.png'
            sheet.paste(Image.open(p), (ci*S, row*(S+16)))
            d.text((ci*S+4, row*(S+16)+S+2), f'case{c}{suf} {n}', fill='black')
        row += 1
out = Path(r'D:/tmp_timing/dose_sheet.png'); sheet.save(out); print(out, sheet.size)
