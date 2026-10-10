# -*- coding: utf-8 -*-
"""LQ 测试集 niqe 标定 + 我们出图在 512 下采样后的排序是否变化 (2026-09-11)
用途: ① 给出"什么都不做"的 niqe 下限 ② 检验"官方若在低分辨率评估会否改变结论"
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

import numpy as np
import torch
import pyiqa
from PIL import Image

TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
PKGS = {
    'M24': Path(r'D:/models/final_compliant/output_dir'),
    'm0s1': Path(r'D:/models/final_compliant_14b/output_dir'),
}
OUT = Path(r'D:/tmp_timing/lq_niqe_calib.json')
N = 30


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    niqe = pyiqa.create_metric('niqe', device=dev)
    res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}

    def measure(name, it):
        vals = []
        t0 = time.time()
        for im in it:
            t = torch.from_numpy(np.asarray(im.convert('RGB'), np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                vals.append(float(niqe(t).item()))
        res[name] = {'niqe': round(float(np.mean(vals)), 4), 'n': len(vals)}
        OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
        print(f'  [{name}] niqe={res[name]["niqe"]:.4f} (n={len(vals)}, {time.time()-t0:.0f}s)',
              flush=True)

    print('=== 4K 原生 ===')
    if 'LQ_full' not in res:
        measure('LQ_full', [Image.open(p) for p in sorted(TEST.glob('*.jpg'))[:N]])
    for tag, d in PKGS.items():
        if f'{tag}_full' in res:
            continue
        measure(f'{tag}_full', [Image.open(p) for p in sorted(d.glob('*.jpg'))[:N]])

    print('=== 下采样到 512 长边 ===')
    if 'LQ_512' not in res:
        measure('LQ_512', [Image.open(p).resize((512, 384)) for p in sorted(TEST.glob('*.jpg'))[:N]])
    for tag, d in PKGS.items():
        if f'{tag}_512' in res:
            continue
        measure(f'{tag}_512', [Image.open(p).resize((512, 384))
                               for p in sorted(d.glob('*.jpg'))[:N]])

    print('\n=== 汇总 ===')
    for k, v in res.items():
        print(f'  {k:<12} niqe={v["niqe"]:.4f}  预测分 {94.444 - 10.028*v["niqe"]:.2f}')


if __name__ == '__main__':
    main()
