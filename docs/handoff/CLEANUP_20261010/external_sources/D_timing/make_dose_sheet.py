from pathlib import Path
from PIL import Image, ImageDraw
N = Path(r'D:/tmp_timing/dose_sweep/native')
names = ['base', 'us_r2_p120', 'us_r2_p300']
rows = []
for c in (13, 2):
    for n in names:
        p = N / f'{n}_case{c}.png'
        rows.append((f'case{c} {n}', Image.open(p) if p.exists() else Image.new('RGB', (512,512), 'red')))
S = 512
sheet = Image.new('RGB', (S*len(names), (S+16)*2), 'white')
d = ImageDraw.Draw(sheet)
for i, (nm, im) in enumerate(rows):
    r, c = divmod(i, len(names))
    sheet.paste(im, (c*S, r*(S+16)))
    d.text((c*S+4, r*(S+16)+S+2), nm, fill='black')
out = Path(r'D:/tmp_timing/dose_sheet.png')
sheet.save(out); print(out, sheet.size)
