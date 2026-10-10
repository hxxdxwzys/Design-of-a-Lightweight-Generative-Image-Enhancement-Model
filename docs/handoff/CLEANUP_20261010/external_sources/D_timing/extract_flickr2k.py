# -*- coding: utf-8 -*-
"""本地解压 Flickr2K.zip -> <out>/flickr_%04d.<原扩展名> (保留原始格式, 不重编码)

用法:
  python extract_flickr2k.py [--zip E:/flickr2k/Flickr2K.zip] [--out D:/datasets/flickr2k] [--rm]
  --rm : 解压成功后删除 zip (省 11G 磁盘)
与云上成功的 download_df2k_zips.sh 一致: 直接落盘原始字节, 仅筛选图片扩展名。
"""
import argparse
import zipfile
from pathlib import Path

EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=r'E:/flickr2k/Flickr2K.zip')
    ap.add_argument('--out', default=r'D:/datasets/flickr2k')
    ap.add_argument('--rm', action='store_true', help='解压成功后删除 zip')
    args = ap.parse_args()

    zpath = Path(args.zip)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert zpath.exists(), f'找不到 {zpath}'
    assert zpath.stat().st_size >= 11600000000, \
        f'zip 未下完 (当前 {zpath.stat().st_size} bytes < 11.6G), 请等下载完成'

    n = 0
    skip = 0
    with zipfile.ZipFile(zpath) as z:
        for nm in z.namelist():
            if nm.endswith('/'):
                continue
            ext = Path(nm).suffix.lower()
            if ext not in EXT:
                continue
            try:
                data = z.read(nm)
                (out / f'flickr_{n:04d}{ext}').write_bytes(data)
                n += 1
                if n % 500 == 0:
                    print(f'  ... {n} 张', flush=True)
            except Exception as e:
                skip += 1
                print(f'[skip] {nm}: {e}', flush=True)

    print(f'[done] 解压 {n} 张, 跳过 {skip} 张, 输出目录 {out}', flush=True)
    if args.rm:
        zpath.unlink()
        print(f'[rm] 已删除 {zpath}', flush=True)


if __name__ == '__main__':
    main()
