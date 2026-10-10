# -*- coding: utf-8 -*-
"""增强变体 GPU 门控 (w48 sweep 后 GPU 空出时运行):
对 M11 混合 100 图 + 6 档变体逐图算 4K clipiqa, 每图选最优 (变体严格更优才换), 输出最终目录。
用法: D:/Python/python.exe enhance_gate.py --base D:/models/cnn_out_mix_s7_m11 --out D:/models/cnn_out_mix_s7_m12
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:/tmp_timing')
import pyiqa  # noqa: E402

ENH = Path(r'D:/models/cnn_out_enh')
VARIANTS = ['sharp1', 'sharp2', 'clahe', 'pop', 'hdr', 'combo']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=r'D:/models/cnn_out_mix_s7_m11')
    ap.add_argument('--out', default=r'D:/models/cnn_out_mix_s7_m12')
    args = ap.parse_args()

    device = 'cuda'
    cq = pyiqa.create_metric('clipiqa', device=device)
    base = Path(args.base)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    files = sorted(base.glob('*.jpg'))
    report = {}
    for i, p in enumerate(files):
        k = p.stem
        a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
        x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).clamp(1e-4, 1 - 1e-4).to(device)
        best_v = float(cq(x).mean().item())
        best_name = 'base'
        best_path = p
        for vn in VARIANTS:
            vp = ENH / vn / f'{k}.jpg'
            if not vp.exists():
                continue
            va = np.asarray(Image.open(vp).convert('RGB')).astype(np.float32) / 255.0
            vx = torch.from_numpy(va).permute(2, 0, 1).unsqueeze(0).clamp(1e-4, 1 - 1e-4).to(device)
            vq = float(cq(vx).mean().item())
            if vq > best_v:
                best_v, best_name, best_path = vq, vn, vp
        shutil.copy2(best_path, out / f'{k}.jpg')
        report[k] = {'used': best_name, 'clipiqa': round(best_v, 5)}
        if (i + 1) % 10 == 0:
            print(f'  [{i + 1}/{len(files)}]', flush=True)

    n_var = sum(1 for v in report.values() if v['used'] != 'base')
    mean = float(np.mean([v['clipiqa'] for v in report.values()]))
    print(f'[summary] 变体采用 {n_var}/100, mean clipiqa={mean:.4f}', flush=True)
    json.dump(report, open(r'D:/tmp_timing/enhance_gate_report.json', 'w'), indent=1)
    print(f'[ok] {out}', flush=True)


if __name__ == '__main__':
    main()
