# -*- coding: utf-8 -*-
"""把 细节能量 与 官方分 的对照从 6 点扩到 12 点 (含 1.4B 家族)"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

DIRS = {
 'M24 sc0.8           43.6951': r'D:/models/final_compliant/output_dir',
 'probeA sc0.4        43.2837': r'D:/models/final_compliant_probeA_sc04/output_dir',
 'sc00 base           37.6316': r'D:/models/final_compliant_M24sc00/output_dir',
 'avg2                36.6414': r'D:/models/final_compliant_M24avg2/output_dir',
 'm0s1 sc0.93 v2      34.0989': r'D:/models/final_compliant_14b/output_dir',
 'g3w18s1 sc0.93      31.8185': r'D:/models/final_compliant_g3w18s1/output_dir',
 'm0s7 sc0.97 v2      31.3942': r'D:/models/wave2_m0s7_s097/output_dir',
 'm0s7 sc0.97 v3      30.9363': r'D:/models/final_compliant_m0s7/output_dir',
 'p4                   7.1829': r'D:/models/final_compliant_M24p4/output_dir',
 'wave2_m0s1_s090        (未交)': r'D:/models/wave2_m0s1_s090/output_dir',
 'wave2_m0s1_s097        (未交)': r'D:/models/wave2_m0s1_s097/output_dir',
 'wave2_m0s1_s100        (未交)': r'D:/models/wave2_m0s1_s100/output_dir',
}
IDX = list(range(1, 101, 5))
res = {}
for k, d in DIRS.items():
    d = Path(d)
    if not d.exists():
        print(f'{k}: DIR MISSING'); continue
    acc = []
    for i in IDX:
        p = d / f'case{i}.jpg'
        if not p.exists(): p = d / f'case{i}.png'
        if not p.exists(): continue
        g = np.asarray(Image.open(p).convert('L'), dtype=np.float32)
        lv = float(ndimage.laplace(g).var())
        lo = ndimage.gaussian_filter(g, 2.0)
        hf = float(np.sqrt(np.mean((g - lo) ** 2)))
        acc.append((lv, hf))
    if not acc:
        print(f'{k}: NO IMAGES'); continue
    a = np.array(acc)
    res[k] = {'n': len(a), 'lapvar': float(a[:,0].mean()), 'hf': float(a[:,2-1].mean())}
    print(f'{k:32s} n={len(a):3d} lapvar={a[:,0].mean():8.1f} hf={a[:,1].mean():6.3f}', flush=True)
Path(r'D:/tmp_timing/detail_all.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
print('DONE')
