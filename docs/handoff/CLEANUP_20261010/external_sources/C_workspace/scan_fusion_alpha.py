#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫描 v3 与 DiffBIR 输出的融合系数 α, 找有参无参最优平衡。

out = α·v3 + (1-α)·DiffBIR,  α=1 纯 v3, α=0 纯 DiffBIR。
在 3 张验证集上评测, 对比官方分校准 (v3=20.27, DiffBIR=18.57)。
"""
import os
import sys
import glob
import numpy as np
from PIL import Image

VAL = 'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
V3_DIR = 'C:/Users/Yuki Nagato/.claude/jobs/0f1d62a5/tmp/v3_raw_pred'
DB_DIR = 'D:/models/diffbir/val4k_out/jpg'
OUT_ROOT = 'D:/models/fusion_scan'


def load_aligned(name):
    """按 case 名加载 v3 和 DiffBIR 输出 (都转成相同尺寸 RGB float32)。"""
    v3f = glob.glob(os.path.join(V3_DIR, name))[0]
    dbf = glob.glob(os.path.join(DB_DIR, name.replace('_lq', '_lq')))[0]
    v3 = np.asarray(Image.open(v3f).convert('RGB'), np.float32)
    db = np.asarray(Image.open(dbf).convert('RGB'), np.float32)
    if v3.shape != db.shape:
        db = np.array(Image.fromarray(db.astype(np.uint8)).resize((v3.shape[1], v3.shape[0])))
    return v3, db


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    # 每张验证图的名字 (从 v3 输出目录)
    v3_files = sorted(glob.glob(os.path.join(V3_DIR, '*.jpg')))
    print(f'v3 输出: {len(v3_files)} 张', flush=True)
    # 生成几个 α 的融合输出
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        adir = os.path.join(OUT_ROOT, f'a{alpha}')
        os.makedirs(adir, exist_ok=True)
        for v3f in v3_files:
            name = os.path.basename(v3f)
            v3, db = load_aligned(name)
            fused = alpha * v3 + (1 - alpha) * db
            Image.fromarray(np.clip(fused, 0, 255).astype(np.uint8)).save(
                os.path.join(adir, name), quality=95)
        print(f'α={alpha} 融合完成 {len(v3_files)} 张', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
