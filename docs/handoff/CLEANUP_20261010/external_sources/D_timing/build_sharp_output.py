# -*- coding: utf-8 -*-
"""自适应锐化输出管线:
规则 (val 校准): s6k输出 clipiqa < 0.55 -> unsharp r=0.6; 0.55~0.65 -> r=0.3; >=0.65 -> 不动
输出: D:/models/cnn_out_s6k_sharp/ (100 张 q98)
然后全量 clipiqa 验证 + 报告聚合增益。
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

S6K = Path(r'D:\models\cnn_out_s6k')
OUT = Path(r'D:\models\cnn_out_s6k_sharp')


def main():
    rep = json.load(open(r'D:\tmp_timing\per_image_report.json'))['test_clipiqa']
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    applied = {'r06': 0, 'r03': 0, 'none': 0}
    for name, v in sorted(rep.items()):
        img = Image.open(S6K / f'{name}.jpg').convert('RGB')
        base = v['s6k']
        if base < 0.55:
            sharp = img.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
            applied['r06'] += 1
        elif base < 0.65:
            sharp = img.filter(ImageFilter.UnsharpMask(radius=2, percent=30, threshold=2))
            applied['r03'] += 1
        else:
            sharp = img
            applied['none'] += 1
        sharp.save(OUT / f'{name}.jpg', quality=98)
    print(f'done {len(rep)} images in {time.time()-t0:.0f}s: {applied}', flush=True)


if __name__ == '__main__':
    main()
