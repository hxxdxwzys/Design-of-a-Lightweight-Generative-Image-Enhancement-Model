# -*- coding: utf-8 -*-
"""补齐 m0s7_s097 + LQ 基线的测试集无参指标 (2026-09-11)
选取方式与既有 nr_test100_results.json 完全一致 (sorted()[:30]) 以保证可比。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PKGS = {
    'm0s7_s097': Path(r'D:/models/wave2_m0s7_s097/output_dir'),
    'LQ_test': Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'),
}
OUT = Path(r'D:/tmp_timing/nr_extra_0911.json')
SAMPLE = 30


def main():
    import numpy as np
    import torch
    import pyiqa
    from PIL import Image

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = {n: pyiqa.create_metric(n, device=dev) for n in ('niqe', 'musiq', 'clipiqa')}
    print('[metric] ready', flush=True)
    res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    for name, d in PKGS.items():
        jpgs = sorted(d.glob('*.jpg'))[:SAMPLE]
        if not jpgs:
            print(f'[skip] {name} 无图'); continue
        vals = {k: [] for k in metrics}
        t0 = time.time()
        for j in jpgs:
            img = Image.open(j).convert('RGB')
            t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(dev)
            for k, m in metrics.items():
                with torch.no_grad():
                    vals[k].append(float(m(t).item()))
        res[name] = {k: round(float(np.mean(v)), 4) for k, v in vals.items()}
        print(f'[{name}] n={len(jpgs)} ({time.time()-t0:.0f}s): ' +
              ' '.join(f'{k}={res[name][k]:.4f}' for k in metrics), flush=True)
        OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('\n=== 汇总 ===')
    for k, v in res.items():
        print(f'  {k:<12} niqe={v["niqe"]:.4f} musiq={v["musiq"]:.4f} clipiqa={v["clipiqa"]:.4f}')
    print(f'\n已存 {OUT}')


if __name__ == '__main__':
    main()
