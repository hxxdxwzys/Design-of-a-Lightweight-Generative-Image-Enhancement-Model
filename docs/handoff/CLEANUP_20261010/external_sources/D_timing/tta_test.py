# -*- coding: utf-8 -*-
"""TTA (几何自集成) 验证 (2026-08-14)
对给定 ckpt 在官方 3-val (4K) 上对比:
  无 TTA  vs  flip/rot 8 向 TTA 平均
指标: lpips + clipiqa (4K, 与官方口径一致)
用法: D:/Python/python.exe tta_test.py --ckpt D:/models/metric_net/net_s6000.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:/tmp_timing')
from train_df2k import NAFNet  # noqa: E402
import pyiqa  # noqa: E402

VAL_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
PATCH, OVERLAP = 512, 128


def make_mask():
    m = torch.ones(PATCH, PATCH)
    ramp = OVERLAP // 2
    for i in range(ramp):
        w = (i + 1) / ramp
        m[i, :] = w
        m[PATCH - 1 - i, :] = w
        m[:, i] = torch.minimum(m[:, i], torch.full((PATCH,), w))
        m[:, PATCH - 1 - i] = torch.minimum(m[:, PATCH - 1 - i], torch.full((PATCH,), w))
    return m


def tiled(net, x, device, mask):
    """x: [1,3,H,W] (0,1) cuda -> [1,3,H,W]"""
    H, W = x.shape[2], x.shape[3]
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
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()

    device = 'cuda'
    net = NAFNet(width=32).to(device).eval()
    ck = torch.load(args.ckpt, map_location=device)
    state = ck['net'] if 'net' in ck else ck
    net.load_state_dict(state)
    print(f'[load] {args.ckpt} step={ck.get("step", "?")}', flush=True)

    lpips_m = pyiqa.create_metric('lpips', device=device)
    cq_m = pyiqa.create_metric('clipiqa', device=device)
    mask = make_mask().to(device)

    for case in (1, 2, 3):
        lq = Image.open(VAL_DIR / f'case{case}_lq.jpg').convert('RGB')
        gt_p = VAL_DIR / f'case{case}_gt.png'
        if not gt_p.exists():
            gt_p = VAL_DIR / f'case{case}_gt.jpg'
        x = load_t(lq).to(device)
        g = load_t(gt_p).to(device).clamp(1e-4, 1 - 1e-4)

        out_plain = tiled(net, x, device, mask)

        # 8 向 TTA
        acc = torch.zeros_like(out_plain)
        n = 0
        for k in range(4):
            for flip in (False, True):
                xt = torch.rot90(x, k, (2, 3))
                if flip:
                    xt = torch.flip(xt, (3,))
                o = tiled(net, xt, device, mask)
                if flip:
                    o = torch.flip(o, (3,))
                o = torch.rot90(o, -k, (2, 3))
                acc += o
                n += 1
        out_tta = acc / n

        for tag, out in [('plain', out_plain), ('tta8', out_tta)]:
            pred = out.clamp(1e-4, 1 - 1e-4)
            lp = float(lpips_m(pred, g).mean().item())
            cq = float(cq_m(pred).mean().item())
            print(f'case{case} {tag}: lpips={lp:.4f} clipiqa={cq:.4f}', flush=True)


if __name__ == '__main__':
    main()
