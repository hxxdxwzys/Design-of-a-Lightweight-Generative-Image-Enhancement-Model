#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v3 + DiffBIR 融合出 100 张测试集, 打包提交 zip。

out = α·v3 + (1-α)·DiffBIR,  α=0.85 (验证集最优: 有参不损失+无参全超)

用法: python fusion_submit.py [--alpha 0.85] [--zip 输出.zip]
"""
import os
import sys
import glob
import shutil
import zipfile
import argparse
import numpy as np
from PIL import Image

V3_DIR = 'D:/models/v3_test100_out'       # v3 出的 100 张测试集
DB_DIR = 'D:/models/diffbir/test100_out'  # DiffBIR 出的 100 张测试集
MODEL_SRC = 'C:/生成式图像增强轻量模型设计挑战/submit_v3/model_dir'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.85)
    ap.add_argument('--zip', default='D:/赛题二_融合增强_4060队_13000000000.zip')
    args = ap.parse_args()
    a = args.alpha

    # 检查 v3 100 张是否存在
    v3_files = sorted(glob.glob(os.path.join(V3_DIR, '*.jpg')))
    db_files = sorted(glob.glob(os.path.join(DB_DIR, '*.png')))
    print(f'v3: {len(v3_files)} 张 | DiffBIR: {len(db_files)} 张', flush=True)
    assert len(v3_files) >= 100, f'v3 输出不足 100 (只有 {len(v3_files)})'
    assert len(db_files) >= 100, f'DiffBIR 输出不足 100 (只有 {len(db_files)})'

    # 建打包目录
    root = os.path.join(os.path.dirname(args.zip), '_pkg')
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, 'output_dir'), exist_ok=True)

    for v3f in v3_files[:100]:
        name = os.path.basename(v3f)  # case1.jpg ... case100.jpg
        v3 = np.asarray(Image.open(v3f).convert('RGB'), np.float32)
        dbf = os.path.join(DB_DIR, name.replace('.jpg', '.png'))
        db = np.asarray(Image.open(dbf).convert('RGB'), np.float32)
        if v3.shape != db.shape:
            db = np.array(Image.fromarray(db.astype(np.uint8)).resize((v3.shape[1], v3.shape[0])))
        fused = a * v3 + (1 - a) * db
        Image.fromarray(np.clip(fused, 0, 255).astype(np.uint8)).save(
            os.path.join(root, 'output_dir', name), quality=95)
    print(f'融合完成 {len(v3_files[:100])} 张 (α={a})', flush=True)

    # model_dir (v3 测速)
    shutil.copytree(MODEL_SRC, os.path.join(root, 'model_dir'))

    # 打包
    with zipfile.ZipFile(args.zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for base, _, files in os.walk(root):
            for f in files:
                fp = os.path.join(base, f)
                zf.write(fp, os.path.relpath(fp, root))
    print(f'zip: {args.zip} ({os.path.getsize(args.zip)/1e6:.0f}MB)', flush=True)

    # 验证
    with zipfile.ZipFile(args.zip) as zf:
        names = zf.namelist()
        assert zf.testzip() is None, 'zip 损坏'
        out_n = sum(1 for n in names if n.endswith('.jpg'))
        mdl = [n.split('/')[-1] for n in names if 'model_dir' in n]
        print(f'验证: output_dir {out_n} 张 ✅ | model_dir {mdl} ✅', flush=True)
    shutil.rmtree(root)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
