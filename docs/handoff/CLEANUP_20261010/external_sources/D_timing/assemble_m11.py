# -*- coding: utf-8 -*-
"""M11 = M10 混合 + HYPIR 逐图择优 (边际>0.005):
HYPIR 赢的图用 hypir_q98_out/result 的 PNG 转存 q98 jpg, 其余保持 M10。
用法: D:/Python/python.exe assemble_m11.py
"""
import json
import shutil
from pathlib import Path

from PIL import Image

M10 = Path(r'D:/models/cnn_out_mix_s7_m10')
HYP = Path(r'D:/models/hypir_q98_out/result')
OUT = Path(r'D:/models/cnn_out_mix_s7_m11')

def main():
    wins = json.load(open(r'D:/tmp_timing/hypir_vs_mix.json'))['hypir_wins_m5']
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for p in sorted(M10.glob('*.jpg')):
        k = p.stem
        if k in wins:
            src = HYP / f'{k}.png'
            assert src.exists(), f'missing {src}'
            Image.open(src).convert('RGB').save(OUT / f'{k}.jpg', quality=98)
        else:
            shutil.copy2(p, OUT / f'{k}.jpg')
    print(f'[ok] {OUT}: HYPIR 换图 {len(wins)} 张', flush=True)


if __name__ == '__main__':
    main()
