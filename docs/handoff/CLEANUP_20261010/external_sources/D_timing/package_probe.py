# -*- coding: utf-8 -*-
"""探测提交打包: HYPIR PNG 输出 → output_dir/case*.jpg (q98) + 1105x 测速模型
用法: python package_probe.py <hypir_result_dir> <out_zip>
"""
import argparse
import os
import shutil
import zipfile
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hypir-dir', default=r'D:\models\hypir_q98_out\result')
    ap.add_argument('--zip', default=r'D:\models\Track2_ProbeHYPIR_Yuki_13000000000.zip')
    ap.add_argument('--model-dir', default=r'D:\models\final_graph\model_dir')
    args = ap.parse_args()

    tmp = Path(r'D:\tmp_timing\probe_out')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    src = Path(args.hypir_dir)
    files = sorted(src.glob('case*.png')) + sorted(src.glob('case*.jpg'))
    assert len(files) >= 100, f'only {len(files)} outputs'
    n = 0
    for p in files:
        stem = p.stem.replace('_lq', '')
        im = Image.open(p).convert('RGB')
        im.save(tmp / f'{stem}.jpg', quality=98)
        n += 1
    print(f'[out] {n} jpgs -> {tmp}', flush=True)

    z = Path(args.zip)
    if z.exists():
        z.unlink()
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(tmp.glob('*.jpg')):
            zf.write(p, f'output_dir/{p.name}')
        zf.write(Path(args.model_dir) / 'your_model.pt', 'model_dir/your_model.pt')
        zf.write(Path(args.model_dir) / 'runner.py', 'model_dir/runner.py')
    print(f'[ok] {z} ({z.stat().st_size/1e6:.0f}MB)', flush=True)


if __name__ == '__main__':
    main()
