# -*- coding: utf-8 -*-
"""VOSR 输出入池: PNG -> q98 JPG 到 sweep_test_out/vosr_05b_os/ (供 compute_choice/assemble 使用)
用法: D:/Python/python.exe vosr_to_pool.py
"""
import shutil
from pathlib import Path

from PIL import Image

SRC = Path(r'C:/models/finals/preset/vosr_test_out')
DST = Path(r'D:/models/sweep_test_out/vosr_05b_os')


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    n = 0
    for p in sorted(SRC.glob('*.png')):
        k = p.stem.replace('_lq', '')
        Image.open(p).convert('RGB').save(DST / f'{k}.jpg', quality=98)
        n += 1
    print(f'[ok] {n} jpgs -> {DST}', flush=True)


if __name__ == '__main__':
    main()
