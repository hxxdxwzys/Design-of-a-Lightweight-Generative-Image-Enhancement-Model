# -*- coding: utf-8 -*-
"""生成 probe_detail 的 100 张 4K 输出 = M24 输出 + detail_boost(sigma=1.0, amount=0.6), q98"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, r'D:/tmp_timing')
from detail_boost import detail_boost

SRC = Path(r'D:/models/final_compliant/output_dir')
DST = Path(r'D:/models/probe_detail/output_dir')
DST.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)

t0 = time.time()
files = sorted(SRC.glob('case*.jpg'), key=lambda p: int(p.stem[4:]))
lv0, lv1 = [], []
for n, p in enumerate(files, 1):
    im = Image.open(p).convert('RGB')
    a = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(a).permute(2, 0, 1)[None]
    with torch.no_grad():
        o = detail_boost(t)[0].permute(1, 2, 0).numpy()
    b = np.clip((o + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(b).save(DST / p.name, quality=98, subsampling=0)
    if n % 20 == 0 or n == len(files):
        print(f'{n}/{len(files)}  {time.time()-t0:.0f}s', flush=True)
print('PROBE_DETAIL_DONE')
