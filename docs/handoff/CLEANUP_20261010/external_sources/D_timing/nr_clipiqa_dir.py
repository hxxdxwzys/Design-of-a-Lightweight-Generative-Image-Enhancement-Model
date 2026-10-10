# -*- coding: utf-8 -*-
"""对某个输出目录的 100 张图算 clipiqa (无参, 不需 GT) — 测试集直证层

用法: D:/Python/python.exe nr_clipiqa_dir.py --dir <output_dir> --out <json>
断点续跑: 已有 json 里算过的图跳过。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--metric', default='clipiqa')
    args = ap.parse_args()

    import numpy as np
    import torch
    import pyiqa
    from PIL import Image

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = Path(args.dir)
    jpgs = sorted(d.glob('*.jpg'))
    if not jpgs:
        print(f'[skip] {d} 无 jpg')
        return
    out = Path(args.out)
    res = json.loads(out.read_text(encoding='utf-8')) if out.exists() else {}
    per = res.setdefault('per', {})
    m = pyiqa.create_metric(args.metric, device=dev)
    t0, done = time.time(), 0
    for j in jpgs:
        if j.stem in per:
            continue
        img = Image.open(j).convert('RGB')
        t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        per[j.stem] = float(m(t).item())
        done += 1
        if done % 10 == 0:
            print(f'  [{done}/{len(jpgs)}] {time.time()-t0:.0f}s', flush=True)
            out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    vals = list(per.values())
    res['n'] = len(vals)
    res['mean'] = float(np.mean(vals))
    res['std'] = float(np.std(vals))
    res['min'] = float(np.min(vals))
    res['max'] = float(np.max(vals))
    res['dir'] = str(d)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'[done] {d.name}: n={res["n"]} mean={res["mean"]:.4f} std={res["std"]:.4f} '
          f'min={res["min"]:.4f} max={res["max"]:.4f} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
