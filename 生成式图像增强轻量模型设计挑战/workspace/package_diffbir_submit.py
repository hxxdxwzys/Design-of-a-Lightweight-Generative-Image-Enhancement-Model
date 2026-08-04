#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DiffBIR 100 张出图打包提交 zip。

用法: python package_diffbir_submit.py [--outdir D:/models/diffbir/test100_out] [--zip 赛题二_xxx.zip]

结构 (官方要求):
  my_work.zip
  ├── output_dir/case1.jpg...case100.jpg   (DiffBIR 出图)
  └── model_dir/your_model.pt + runner.py  (v3 测速, 复用 submit_v3)
"""
import os
import sys
import glob
import shutil
import zipfile
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='D:/models/diffbir/test100_out')
    ap.add_argument('--zip', default='D:/赛题二_DiffBIR增强_4060队_13000000000.zip')
    ap.add_argument('--model_dir_src', default='C:/生成式图像增强轻量模型设计挑战/submit_v3/model_dir')
    args = ap.parse_args()

    # 1. 检查出图完整性
    pngs = sorted(glob.glob(os.path.join(args.outdir, '*.png')))
    print(f'出图数: {len(pngs)}')
    assert len(pngs) >= 100, f'出图不足 100 (只有 {len(pngs)})'

    # 2. 建打包目录
    root = os.path.join(os.path.dirname(args.zip), '_pkg')
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, 'output_dir'), exist_ok=True)

    # 3. png -> jpg (按官方命名 case1.jpg...case100.jpg)
    for i, p in enumerate(pngs[:100]):
        import numpy as np
        from PIL import Image
        img = Image.open(p).convert('RGB')
        name = os.path.basename(p).rsplit('.', 1)[0] + '.jpg'
        img.save(os.path.join(root, 'output_dir', name), quality=95)
    print(f'output_dir: {len(os.listdir(os.path.join(root, "output_dir")))} 张 jpg')

    # 4. model_dir (v3 测速)
    shutil.copytree(args.model_dir_src, os.path.join(root, 'model_dir'))
    print('model_dir: v3 测速 (your_model.pt + runner.py)')

    # 5. 打包 zip
    with zipfile.ZipFile(args.zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for base, _, files in os.walk(root):
            for f in files:
                fp = os.path.join(base, f)
                arc = os.path.relpath(fp, root)
                zf.write(fp, arc)
    sz = os.path.getsize(args.zip) / 1e6
    print(f'zip 完成: {args.zip} ({sz:.0f}MB)')

    # 6. 验证
    with zipfile.ZipFile(args.zip) as zf:
        names = zf.namelist()
        assert zf.testzip() is None, 'zip 损坏!'
        out_n = sum(1 for n in names if '/output_dir/' in n and n.endswith('.jpg'))
        mdl = [n.split('/')[-1] for n in names if '/model_dir/' in n]
        print(f'验证: output_dir {out_n} 张 ✅ | model_dir {mdl} ✅ | 无损坏 ✅')
    shutil.rmtree(root)
    print('ALL DONE')


if __name__ == '__main__':
    main()
