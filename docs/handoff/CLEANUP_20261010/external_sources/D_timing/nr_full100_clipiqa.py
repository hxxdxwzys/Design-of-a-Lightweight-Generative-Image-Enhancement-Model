# -*- coding: utf-8 -*-
"""全 100 张 clipiqa 对比 + per-image 分析 (2026-09-09)
M24 vs m0s1: 逐图 clipiqa -> 胜负统计 + 分布 + 崩溃图识别
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

PKGS = {
    'M24': Path(r'D:/models/final_compliant/output_dir'),
    'm0s1': Path(r'D:/models/final_compliant_14b/output_dir'),
}
OUT = Path(r'D:/tmp_timing/nr_full100_clipiqa.json')


def main():
    import pyiqa
    from PIL import Image
    dev = 'cuda'
    m = pyiqa.create_metric('clipiqa', device=dev)
    res = {}
    for pkg, d in PKGS.items():
        jpgs = sorted(d.glob('*.jpg'))
        vals = {}
        t0 = time.time()
        for j in jpgs:
            img = Image.open(j).convert('RGB')
            t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                vals[j.stem] = round(float(m(t).item()), 4)
        res[pkg] = vals
        v = np.array(list(vals.values()))
        print(f'[{pkg}] {len(vals)} 张 ({time.time()-t0:.0f}s): mean={v.mean():.4f} '
              f'std={v.std():.4f} min={v.min():.4f} max={v.max():.4f}', flush=True)
    # 逐图对比
    common = sorted(set(res['M24']) & set(res['m0s1']))
    wins_m24 = [k for k in common if res['M24'][k] > res['m0s1'][k]]
    wins_m0s1 = [k for k in common if res['m0s1'][k] > res['M24'][k]]
    print(f'\n逐图胜负: M24 胜 {len(wins_m24)} / m0s1 胜 {len(wins_m0s1)} (共 {len(common)})')
    diffs = np.array([res['M24'][k] - res['m0s1'][k] for k in common])
    print(f'M24-m0s1 clipiqa diff: mean={diffs.mean():+.4f} min={diffs.min():+.4f} max={diffs.max():+.4f}')
    # 最大崩图
    worst = sorted(common, key=lambda k: res['M24'][k] - res['m0s1'][k], reverse=True)[:8]
    print('\nm0s1 相对最差的 8 张:')
    for k in worst:
        print(f'  {k}: M24={res["M24"][k]:.4f} m0s1={res["m0s1"][k]:.4f}')
    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'\n已存: {OUT}')


if __name__ == '__main__':
    main()
