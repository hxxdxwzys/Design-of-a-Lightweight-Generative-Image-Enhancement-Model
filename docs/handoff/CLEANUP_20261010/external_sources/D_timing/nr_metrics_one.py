# -*- coding: utf-8 -*-
"""测试集无参指标: 单包模式 (补跑 M24 基准)
用法: python nr_metrics_one.py <pkg名>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT_ROOT = Path(r'D:/models')
PKGS = {
    'M24': OUT_ROOT / 'final_compliant/output_dir',
    'm0s1': OUT_ROOT / 'final_compliant_14b/output_dir',
    'g3w18s1': OUT_ROOT / 'final_compliant_g3w18s1/output_dir',
    'm0s7': OUT_ROOT / 'final_compliant_m0s7/output_dir',
}
RES = Path(r'D:/tmp_timing/nr_test100_results.json')
SAMPLE = 30


def main():
    import pyiqa
    from PIL import Image
    pkg = sys.argv[1]
    d = PKGS[pkg]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = {k: pyiqa.create_metric(k, device=dev) for k in ('clipiqa', 'musiq', 'niqe')}
    jpgs = sorted(d.glob('*.jpg'))[:SAMPLE]
    vals = {k: [] for k in metrics}
    t0 = time.time()
    for j in jpgs:
        img = Image.open(j).convert('RGB')
        t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        for k, m in metrics.items():
            with torch.no_grad():
                vals[k].append(float(m(t).item()))
    r = {k: round(float(sum(v) / len(v)), 4) for k, v in vals.items()}
    print(f'[{pkg}] {len(jpgs)} 张 ({time.time()-t0:.0f}s): ' + ' '.join(f'{k}={r[k]:.4f}' for k in r), flush=True)
    old = json.loads(RES.read_text(encoding='utf-8')) if RES.exists() else {}
    old[pkg] = r
    RES.write_text(json.dumps(old, indent=1), encoding='utf-8')
    print('saved', RES)


if __name__ == '__main__':
    main()
