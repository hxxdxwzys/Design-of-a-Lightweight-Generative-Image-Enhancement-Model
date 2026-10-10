# -*- coding: utf-8 -*-
"""通用合规打包 (2026-09-08): 任意 output_dir + model_dir -> zip
用法: python package_generic.py --name <zip名不含.zip> --outdir <output_dir> --modeldir <model_dir>
输出: D:/models/<name>.zip (纯 ASCII 命名由调用方保证)
"""
import argparse
import shutil
import zipfile
from pathlib import Path

FIXED_TIME = (2026, 9, 1, 0, 0, 0)
EXCLUDE_SUFFIX = {'.pyc', '.log', '.tmp', '.npy', '.png'}
EXCLUDE_NAME = {'__pycache__'}


def add_file(zf, fpath, arcname):
    zi = zipfile.ZipInfo(arcname, FIXED_TIME)
    zi.compress_type = zipfile.ZIP_DEFLATED
    with open(fpath, 'rb') as f:
        zf.writestr(zi, f.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--outdir', required=True, help='output_dir 源 (100 jpg)')
    ap.add_argument('--modeldir', required=True, help='model_dir 源')
    args = ap.parse_args()
    outdir = Path(args.outdir)
    modeldir = Path(args.modeldir)
    out_zip = Path(r'D:/models') / f'{args.name}.zip'
    if out_zip.exists():
        out_zip.unlink()
    jpgs = sorted(outdir.glob('*.jpg'))
    assert len(jpgs) == 100, f'output_dir 需 100 jpg, 实际 {len(jpgs)}'
    with zipfile.ZipFile(out_zip, 'w') as zf:
        for j in jpgs:
            add_file(zf, j, f'output_dir/{j.name}')
        for p in sorted(modeldir.rglob('*')):
            if not p.is_file():
                continue
            if p.suffix in EXCLUDE_SUFFIX or any(s in p.parts for s in EXCLUDE_NAME):
                continue
            arc = 'model_dir/' + str(p.relative_to(modeldir)).replace('\\', '/')
            add_file(zf, p, arc)
    sz = out_zip.stat().st_size / 1e9
    # 校验
    with zipfile.ZipFile(out_zip) as z:
        names = z.namelist()
        outs = [n for n in names if n.startswith('output_dir/')]
        mds = [n for n in names if n.startswith('model_dir/')]
        assert len(outs) == 100
        assert 'model_dir/runner.py' in mds and 'model_dir/your_model.pt' in mds
    print(f'[save] {out_zip} ({sz:.2f} GB) | output {len(outs)} | model_dir {len(mds)}')
    print('PACK_PASS')


if __name__ == '__main__':
    main()
