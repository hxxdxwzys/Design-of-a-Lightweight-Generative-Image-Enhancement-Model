#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""细化 α ∈ {0.82, 0.87} 融合 (0.8-0.9 之间找最优)。"""
import os
import glob
import numpy as np
from PIL import Image

V3 = 'C:/Users/Yuki Nagato/.claude/jobs/0f1d62a5/tmp/v3_raw_pred'
DB = 'D:/models/diffbir/val4k_out/jpg'
OUT_ROOT = 'D:/models/fusion_scan'


def main():
    for a in [0.82, 0.87]:
        adir = os.path.join(OUT_ROOT, f'a{a}')
        os.makedirs(adir, exist_ok=True)
        for v3f in sorted(glob.glob(os.path.join(V3, '*.jpg'))):
            name = os.path.basename(v3f)
            v3 = np.asarray(Image.open(v3f).convert('RGB'), np.float32)
            db = np.asarray(Image.open(os.path.join(DB, name)).convert('RGB'), np.float32)
            if v3.shape != db.shape:
                db = np.array(Image.fromarray(db.astype(np.uint8)).resize((v3.shape[1], v3.shape[0])))
            fused = a * v3 + (1 - a) * db
            Image.fromarray(np.clip(fused, 0, 255).astype(np.uint8)).save(
                os.path.join(adir, name), quality=95)
        print(f'α={a} 完成', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
