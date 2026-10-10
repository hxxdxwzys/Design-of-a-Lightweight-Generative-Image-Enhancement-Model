# -*- coding: utf-8 -*-
"""B: 细节剂量扫描 (2026-09-13)
目标函数换成 lapvar (与官方分 ρ=0.93 的那个量), 不再是 niqe/clipiqa。
对 M24 的 100 张 4K 图施加不同强度的确定性细节增强, 测 lapvar/hf + 存对照裁切图。
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

SRC = Path(r'D:/models/final_compliant/output_dir')
OUT = Path(r'D:/tmp_timing/dose_sweep')
OUT.mkdir(parents=True, exist_ok=True)
IDX = list(range(1, 101, 5))          # 与之前测量同 20 张, 可直比
NATIVE = OUT / 'native'
NATIVE.mkdir(exist_ok=True)


def lapvar_gray(g):
    return float(ndimage.laplace(g).var())


def hf_gray(g):
    return float(np.sqrt(np.mean((g - ndimage.gaussian_filter(g, 2.0)) ** 2)))


def measure(im):
    g = np.asarray(im.convert('L'), dtype=np.float32)
    return lapvar_gray(g), hf_gray(g)


def us(im, r, pct, thr=0):
    return im.filter(ImageFilter.UnsharpMask(radius=r, percent=pct, threshold=thr))


VARIANTS = {
    'base':        lambda im: im,
    'us_r1_p60':   lambda im: us(im, 1.0, 60),
    'us_r1_p120':  lambda im: us(im, 1.0, 120),
    'us_r2_p60':   lambda im: us(im, 2.0, 60),
    'us_r2_p120':  lambda im: us(im, 2.0, 120),
    'us_r2_p200':  lambda im: us(im, 2.0, 200),
    'us_r3_p150':  lambda im: us(im, 3.0, 150),
    'us_r2_p300':  lambda im: us(im, 2.0, 300),
    'us_r4_p300':  lambda im: us(im, 4.0, 300),
}

rows = []
for name, fn in VARIANTS.items():
    lv, hf = [], []
    for i in IDX:
        im = Image.open(SRC / f'case{i}.jpg').convert('RGB')
        o = fn(im)
        a, b = measure(o)
        lv.append(a); hf.append(b)
        if name in ('base', 'us_r2_p120', 'us_r2_p300') and i in (13, 2):
            w, h = o.size
            o.crop((1500, 1200, 1500+512, 1200+512)).save(NATIVE / f'{name}_case{i}.png')
    m = (float(np.mean(lv)), float(np.mean(hf)))
    rows.append((name, m[0], m[1], m[0]/762.4))
    print(f'{name:12s} lapvar={m[0]:8.1f}  hf={m[1]:6.3f}  x{m[0]/762.4:5.2f} vs M24', flush=True)

print('\nM24 基准 lapvar=762.4 (q98), 测试集 LQ lapvar=70.7')
Path(OUT / 'sweep.json').write_text(json.dumps(
    [{'name': n, 'lapvar': l, 'hf': h, 'ratio': r} for n, l, h, r in rows], indent=1), encoding='utf-8')
print('DONE')
