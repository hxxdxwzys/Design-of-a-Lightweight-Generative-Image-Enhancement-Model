# -*- coding: utf-8 -*-
"""打包提交 zip v2 (2026-08-14): 固定 ZipInfo 时间戳 + ASCII 命名 + 结构校验
用法:
  D:/Python/python.exe package_submit_v2.py --out-dir D:/models/sweep_test_out/f2k_s5000 \
      --model-dir D:/models/final_graph7 --name MetricV8 --account Remay
  -> D:/models/Track2_MetricV8_Remay_13000000000.zip
"""
import argparse
import zipfile
from pathlib import Path

FIXED_TS = (2024, 1, 1, 0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--model-dir', default=r'D:/models/final_graph7')
    ap.add_argument('--name', required=True, help='作品名 (纯 ASCII)')
    ap.add_argument('--account', default='Remay')
    ap.add_argument('--phone', default='13000000000')
    ap.add_argument('--zip', default=None)
    args = ap.parse_args()

    out, mdl = Path(args.out_dir), Path(args.model_dir)
    assert out.is_dir() and mdl.is_dir(), 'dirs must exist'
    jpgs = sorted(out.glob('*.jpg'))
    assert len(jpgs) >= 100, f'output has only {len(jpgs)} jpgs'
    assert (mdl / 'your_model.pt').exists() and (mdl / 'runner.py').exists(), 'model_dir 缺文件'
    for p in jpgs:
        assert p.name.startswith('case') and p.name.endswith('.jpg'), f'bad name {p.name}'

    z = Path(args.zip) if args.zip else Path(rf'D:/models/Track2_{args.name}_{args.account}_{args.phone}.zip')
    if z.exists():
        z.unlink()
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in jpgs:
            zi = zipfile.ZipInfo(f'output_dir/{p.name}', date_time=FIXED_TS)
            zi.compress_type = zipfile.ZIP_DEFLATED
            with open(p, 'rb') as f:
                zf.writestr(zi, f.read())
        for name in ('your_model.pt', 'runner.py'):
            zi = zipfile.ZipInfo(f'model_dir/{name}', date_time=FIXED_TS)
            zi.compress_type = zipfile.ZIP_DEFLATED
            with open(mdl / name, 'rb') as f:
                zf.writestr(zi, f.read())
    print(f'[ok] {z} ({z.stat().st_size/1e6:.0f}MB, {len(jpgs)} jpgs)', flush=True)


if __name__ == '__main__':
    main()
