# -*- coding: utf-8 -*-
"""预切 512x512 HQ patch 缓存 (JPEG q95) -> 训练时快速解码 (PNG 解码是瓶颈: 顺序 4 张/步 ~400ms)"""
from __future__ import annotations
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

HQ_DIRS = [Path(r'D:/datasets/div2k/pngtxt_ori'), Path(r'D:/datasets/flickr2k')]
DST = Path(r'D:/datasets/tiny_patches')
N = 20000
RES = 512
DST.mkdir(parents=True, exist_ok=True)

files = []
for d in HQ_DIRS:
    files += sorted(d.glob('*.png'))
print(f'[src] {len(files)} png', flush=True)
rng = np.random.RandomState(7)
dst_list = [DST / f'p{i:06d}.jpg' for i in range(N)]
todo = [i for i, p in enumerate(dst_list) if not p.exists()]
print(f'[todo] {len(todo)} 张待切', flush=True)

def work(i):
    for _ in range(6):
        p = files[rng.randint(len(files))]
        try:
            with Image.open(p) as im:
                im = im.convert('RGB')
                W, H = im.size
                if W < RES or H < RES:
                    return False
                x = rng.randint(0, W - RES + 1); y = rng.randint(0, H - RES + 1)
                im.crop((x, y, x + RES, y + RES)).save(dst_list[i], quality=95)
            return True
        except Exception:
            continue
    return False

t0 = time.time(); ok = 0
with ThreadPoolExecutor(max_workers=10) as ex:
    for n, r in enumerate(ex.map(work, todo), 1):
        ok += bool(r)
        if n % 2000 == 0:
            print(f'  {n}/{len(todo)} ok={ok} {(time.time()-t0)/n*1000:.0f}ms/张', flush=True)
print(f'[done] {ok} 张, {(time.time()-t0)/60:.1f} min -> {DST}', flush=True)
print('PREP_PATCHES_DONE')
