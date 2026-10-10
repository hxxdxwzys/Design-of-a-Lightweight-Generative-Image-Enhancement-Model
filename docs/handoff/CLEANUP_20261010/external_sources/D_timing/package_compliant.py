# -*- coding: utf-8 -*-
"""合规提交打包 (2026-09-03): output_dir 复用 M24 的 100 张 + 自包含 model_dir
输出: D:/models/Track2_CompliantM24_Remay_13000000000.zip
"""
import os
import shutil
import sys
import zipfile
from pathlib import Path

M24_ZIP = r'C:/生成式图像增强轻量模型设计挑战/提交历史/Track2_MetricMixS7M24——256.3045.zip'
MD = Path(r'D:/models/final_compliant/model_dir')
OUT = Path(r'D:/models/Track2_CompliantM24_Remay_13000000000.zip')
TMP = Path(r'D:/tmp_timing/pack_tmp')

FIXED_TIME = (2026, 9, 1, 0, 0, 0)
EXCLUDE_SUFFIX = {'.pyc', '.log', '.tmp', '.npy'}
EXCLUDE_NAME = {'__pycache__'}


def add_file(zf, fpath, arcname):
    zi = zipfile.ZipInfo(arcname, FIXED_TIME)
    zi.compress_type = zipfile.ZIP_DEFLATED
    with open(fpath, 'rb') as f:
        zf.writestr(zi, f.read())


def main():
    if OUT.exists():
        OUT.unlink()
    TMP.mkdir(parents=True, exist_ok=True)
    # 1) 提取 M24 output_dir
    out_src = TMP / 'output_dir'
    if out_src.exists():
        shutil.rmtree(out_src)
    out_src.mkdir(parents=True)
    with zipfile.ZipFile(M24_ZIP) as z:
        for n in z.namelist():
            if n.startswith('output_dir/') and n.endswith('.jpg'):
                (out_src / n.split('/')[-1]).write_bytes(z.read(n))
    n_jpg = len(list(out_src.glob('*.jpg')))
    print(f'output_dir: {n_jpg} 张', flush=True)
    assert n_jpg == 100
    # 2) 打包
    with zipfile.ZipFile(OUT, 'w') as zf:
        for j in sorted(out_src.glob('*.jpg')):
            add_file(zf, j, f'output_dir/{j.name}')
        # model_dir 递归 (排除缓存/临时)
        for p in sorted(MD.rglob('*')):
            if not p.is_file():
                continue
            if p.suffix in EXCLUDE_SUFFIX or any(s in p.parts for s in EXCLUDE_NAME):
                continue
            if p.name == 'your_model.pt' or p.name == 'runner.py' or p.parent != MD:
                arc = 'model_dir/' + str(p.relative_to(MD)).replace('\\', '/')
                add_file(zf, p, arc)
            else:
                arc = 'model_dir/' + str(p.relative_to(MD)).replace('\\', '/')
                add_file(zf, p, arc)
    sz = OUT.stat().st_size / 1e9
    print(f'[save] {OUT} ({sz:.2f} GB)', flush=True)
    # 3) 结构校验
    with zipfile.ZipFile(OUT) as z:
        names = z.namelist()
        outs = [n for n in names if n.startswith('output_dir/')]
        mds = [n for n in names if n.startswith('model_dir/')]
        print(f'output_dir: {len(outs)} | model_dir: {len(mds)}', flush=True)
        assert len(outs) == 100
        assert 'model_dir/runner.py' in mds and 'model_dir/your_model.pt' in mds
    print('PACK_PASS', flush=True)


if __name__ == '__main__':
    main()
