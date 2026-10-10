# -*- coding: utf-8 -*-
"""逐图超感知增强变体生成 (CPU, 不占 GPU):
对 M10 混合目录的 100 张 4K 图生成 6 档 aggressive 变体:
  sharp1(轻度锐化) / sharp2(重度锐化) / clahe(局部对比) / pop(对比+饱和) / hdr(S曲线) / combo(锐化+对比+饱和)
输出 D:/models/cnn_out_enh/<variant>/caseN.jpg (q98)。
之后用 GPU 跑 4K clipiqa 逐图门控 (enhance_gate.py), 只在更优时替换。
用法: D:/Python/python.exe make_enhance_variants.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SRC = Path(r'D:/models/cnn_out_mix_s7_m10')
OUT = Path(r'D:/models/cnn_out_enh')

KERNELS = {
    'sharp1': (3, 1.2, 50),
    'sharp2': (6, 2.0, 90),
}


def unsharp(a, radius, amount, threshold):
    im = Image.fromarray(a)
    im = im.filter(ImageFilter.UnsharpMask(radius=radius, percent=amount, threshold=threshold))
    return np.asarray(im)


def clahe_y(a):
    # 简易 CLAHE: 对亮度通道做 64x64 tile 直方图均衡 + 原图 0.6 混合
    im = Image.fromarray(a)
    y = im.convert('L')
    y_eq = ImageOps.equalize(y)
    y_mix = Image.blend(y, y_eq, 0.6)
    return np.asarray(Image.merge('RGB', (y_mix, y_mix, y_mix)))


def make_variants(a):
    im = Image.fromarray(a)
    v = {}
    v['sharp1'] = np.asarray(im.filter(ImageFilter.UnsharpMask(radius=3, percent=120, threshold=40)))
    v['sharp2'] = np.asarray(im.filter(ImageFilter.UnsharpMask(radius=6, percent=200, threshold=80)))
    v['clahe'] = clahe_y(a)
    v['pop'] = np.asarray(ImageEnhance.Color(ImageEnhance.Contrast(im).enhance(1.2)).enhance(1.25))
    v['hdr'] = np.asarray(ImageOps.autocontrast(im, cutoff=2))
    v['combo'] = np.asarray(ImageEnhance.Color(
        ImageEnhance.Contrast(im.filter(ImageFilter.UnsharpMask(radius=5, percent=160, threshold=60))).enhance(1.15)
    ).enhance(1.2))
    return v


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob('*.jpg'))
    print(f'[start] {len(files)} images', flush=True)
    for i, p in enumerate(files):
        a = np.asarray(Image.open(p).convert('RGB'))
        for name, arr in make_variants(a).items():
            d = OUT / name
            d.mkdir(exist_ok=True)
            Image.fromarray(arr).save(d / p.name, quality=98)
        if (i + 1) % 10 == 0:
            print(f'  [{i + 1}/{len(files)}]', flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
