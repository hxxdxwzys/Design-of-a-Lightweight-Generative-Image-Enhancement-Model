# -*- coding: utf-8 -*-
"""val 3 对流形门控 (2026-08-17): 计算某输出目录在官方 3-val 上的 clipiqa + lpips(vs GT)
用法: D:/Python/python.exe val_gate.py --src <val 输出目录> [--out <json>]
输出: JSON {"clip": .., "lpips": .., "pass": bool, "per": {caseN: {...}}}
门控: clipiqa>=0.55 且 lpips<=0.372 (VOSR 0.629/0.342 放宽阈值)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:/tmp_timing')
import pyiqa  # noqa: E402

VAL_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')  # 默认, 可被 --val-dir 覆盖


def load_tensor(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).clamp(1e-4, 1 - 1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--clip-min', type=float, default=0.55, help='clipiqa 门控下限 (1.4B 前沿扫描用 0.919)')
    ap.add_argument('--lpips-max', type=float, default=0.372, help='lpips 门控上限 (1.4B 前沿扫描用 0.360)')
    ap.add_argument('--val-dir', default=r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
    args = ap.parse_args()

    device = 'cuda'
    clipiqa_m = pyiqa.create_metric('clipiqa', device=device)
    lpips_m = pyiqa.create_metric('lpips', device=device)
    VAL_DIR = Path(args.val_dir)

    src = Path(args.src)
    per = {}
    for gtp in sorted(VAL_DIR.glob('*_gt.*')):
        name = gtp.stem.replace('_gt', '')
        cands = [p for p in src.iterdir() if p.stem.startswith(name) and '_lq' in p.stem]
        if not cands:
            print(f'[warn] {name}: 无对应输出', flush=True)
            continue
        out_t = load_tensor(cands[0]).to(device)
        gt_t = load_tensor(gtp).to(device)
        with torch.no_grad():
            cq = float(clipiqa_m(out_t).mean().item())
            lp = float(lpips_m(out_t, gt_t).mean().item())
        per[name] = {'clipiqa': round(cq, 4), 'lpips': round(lp, 4)}
        print(f'[val] {name}: clipiqa={cq:.4f} lpips={lp:.4f}', flush=True)

    res = {
        'clip': round(float(np.mean([v['clipiqa'] for v in per.values()])), 4) if per else 0.0,
        'lpips': round(float(np.mean([v['lpips'] for v in per.values()])), 4) if per else 9.9,
        'pass': bool(per) and bool(np.mean([v['clipiqa'] for v in per.values()]) >= args.clip_min
                       and np.mean([v['lpips'] for v in per.values()]) <= args.lpips_max),
        'per': per,
    }
    print(f'[gate] clip={res["clip"]} lpips={res["lpips"]} pass={res["pass"]}', flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
