# -*- coding: utf-8 -*-
"""训练前校验数据目录里的图片都可被 PIL 正常打开 (防 Flickr2K 个别损坏图让 train 崩溃)。
用法: python validate_imgs.py [--dir D:/datasets/flickr2k] [--dir ...]  (可多个 --dir)
坏图移到 <dir>/_corrupt/ (不删, 可复查)。
"""
import argparse
from pathlib import Path

from PIL import Image

EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', action='append', default=[], help='可重复, 校验多个目录')
    args = ap.parse_args()
    dirs = args.dir or [r'D:/datasets/flickr2k', r'D:/datasets/div2k/pngtxt_ori']

    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f'[skip] {d} 不存在', flush=True)
            continue
        imgs = [p for p in sorted(d.iterdir()) if p.suffix.lower() in EXT]
        bad = 0
        ok = 0
        for p in imgs:
            try:
                with Image.open(p) as im:
                    im.verify()
                # verify 后需重新 open 才能 convert
                with Image.open(p) as im:
                    im.convert('RGB').load()
                ok += 1
            except Exception as e:
                bad += 1
                cdir = d / '_corrupt'
                cdir.mkdir(exist_ok=True)
                p.rename(cdir / p.name)
                print(f'[bad] {p.name}: {e}', flush=True)
        print(f'[done] {d.name}: {ok} ok, {bad} bad (moved to _corrupt)', flush=True)


if __name__ == '__main__':
    main()
