# -*- coding: utf-8 -*-
"""全 ckpt val sweep: 对 --ckpt-dir 下所有 net_s*.pt 渲染 3 张 val 并算 lpips/clipiqa,
输出排序表 (找最优 ckpt)。公式质量分量参照 net_infer.py 口径。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:\tmp_timing')
from train_df2k import NAFNet  # noqa: E402
import pyiqa  # noqa: E402

VAL_DIR = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\验证集')
PATCH = 512
OVERLAP = 128


def make_mask(patch=512, overlap=128):
    m = torch.ones(patch, patch)
    ramp = overlap // 2
    for i in range(ramp):
        w = (i + 1) / ramp
        m[i, :] = w
        m[patch - 1 - i, :] = w
        m[:, i] = torch.minimum(m[:, i], torch.full((patch,), w))
        m[:, patch - 1 - i] = torch.minimum(m[:, patch - 1 - i], torch.full((patch,), w))
    return m


def tiled_infer(net, img_pil, device):
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    mask = make_mask(PATCH, OVERLAP).to(device)
    acc = torch.zeros(1, 3, H, W, device=device)
    wsum = torch.zeros(1, 1, H, W, device=device)
    step = PATCH - OVERLAP
    ys = list(range(0, max(H - PATCH, 0) + 1, step))
    xs = list(range(0, max(W - PATCH, 0) + 1, step))
    if (H - PATCH) > ys[-1]:
        ys.append(H - PATCH)
    if (W - PATCH) > xs[-1]:
        xs.append(W - PATCH)
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                p = x[:, :, y0:y0 + PATCH, x0:x0 + PATCH]
                out = net(p).clamp(0, 1)
                acc[:, :, y0:y0 + PATCH, x0:x0 + PATCH] += out * mask
                wsum[:, :, y0:y0 + PATCH, x0:x0 + PATCH] += mask
    return acc / wsum


def load_t(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = 'cuda'
    lpips_m = pyiqa.create_metric('lpips', device=device)
    cq_m = pyiqa.create_metric('clipiqa', device=device)
    net = NAFNet(width=args.width).to(device).eval()

    ckpts = sorted(Path(args.ckpt_dir).glob('net_s*.pt'),
                   key=lambda p: int(p.stem.split('s')[-1]))
    print(f'[sweep] {len(ckpts)} checkpoints, width={args.width}', flush=True)

    # 预加载 GT
    gts = {}
    for case in (1, 2, 3):
        p = VAL_DIR / f'case{case}_gt.png'
        if not p.exists():
            p = VAL_DIR / f'case{case}_gt.jpg'
        gts[case] = load_t(p).to(device).clamp(1e-4, 1 - 1e-4)

    results = []
    for ck in ckpts:
        net.load_state_dict(torch.load(ck, map_location=device)['net'])
        net.eval()
        lp, cq = [], []
        with torch.no_grad():
            for case in (1, 2, 3):
                lq = Image.open(VAL_DIR / f'case{case}_lq.jpg').convert('RGB')
                out = tiled_infer(net, lq, device)
                pred = out.clamp(1e-4, 1 - 1e-4)
                lp.append(float(lpips_m(pred, gts[case]).mean().item()))
                cq.append(float(cq_m(pred).mean().item()))
        mlp, mcq = np.mean(lp), np.mean(cq)
        q = 13.5 + 32 * (0.4262 - mlp) + 40 * (mcq - 0.2509)
        results.append({'ckpt': ck.name, 'lpips': mlp, 'clipiqa': mcq, 'q': q})
        print(f'  {ck.name}: lpips={mlp:.4f} clipiqa={mcq:.4f} Q={q:.2f}', flush=True)

    results.sort(key=lambda r: -r['q'])
    print('\n=== sorted by Q (best first) ===', flush=True)
    for r in results[:10]:
        print(f'  {r["ckpt"]}: lpips={r["lpips"]:.4f} clipiqa={r["clipiqa"]:.4f} Q={r["q"]:.2f}', flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'[saved] {args.out}', flush=True)


if __name__ == '__main__':
    main()
