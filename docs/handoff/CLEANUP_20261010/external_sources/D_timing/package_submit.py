# -*- coding: utf-8 -*-
"""打包提交 zip: output_dir(100 jpg) + model_dir(your_model.pt + runner.py)
用法: python package_submit.py --out-dir <output_dir> --model-dir <model_dir> --zip D:/models/Track2_xxx.zip
"""
import argparse
import os
import shutil
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--zip', required=True)
    args = ap.parse_args()

    out, mdl, z = Path(args.out_dir), Path(args.model_dir), Path(args.zip)
    assert out.is_dir() and mdl.is_dir(), 'dirs must exist'
    assert z.suffix == '.zip'
    n = len(list(out.glob('*.jpg')))
    assert n >= 100, f'output has only {n} jpgs'
    assert (mdl / 'your_model.pt').exists() and (mdl / 'runner.py').exists()

    if z.exists():
        z.unlink()
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(out.glob('*.jpg')):
            zf.write(p, f'output_dir/{p.name}')
        zf.write(mdl / 'your_model.pt', 'model_dir/your_model.pt')
        zf.write(mdl / 'runner.py', 'model_dir/runner.py')
    size = z.stat().st_size / 1e6
    print(f'[ok] {z} ({size:.0f}MB, {n} jpgs)', flush=True)


if __name__ == '__main__':
    main()
