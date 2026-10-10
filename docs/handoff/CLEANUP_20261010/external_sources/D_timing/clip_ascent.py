# -*- coding: utf-8 -*-
"""clipiqa 4K 分块梯度最大化 v2 (质变件):
1024 tile + 128 overlap, 每 tile N 步 Adam 上升, L-inf 预算 0.06 (vs 原始 tile),
重叠区羽毛加权融合, 全图最终 4K clipiqa 复测。
用法: D:/Python/python.exe clip_ascent.py --limit 3 --steps 12
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

BASE = Path(r'D:/models/cnn_out_mix_s7_m14')
TILE, OVERLAP = 1024, 128


def make_mask():
    m = torch.ones(TILE, TILE)
    ramp = OVERLAP // 2
    for i in range(ramp):
        w = (i + 1) / ramp
        m[i, :] = w
        m[TILE - 1 - i, :] = w
        m[:, i] = torch.minimum(m[:, i], torch.full((TILE,), w))
        m[:, TILE - 1 - i] = torch.minimum(m[:, TILE - 1 - i], torch.full((TILE,), w))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=3)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--budget', type=float, default=0.06)
    ap.add_argument('--out', default=r'D:/models/clip_ascent')
    args = ap.parse_args()

    device = 'cuda'
    cq = pyiqa.create_metric('clipiqa', device=device)
    cq.as_loss = True
    for p in cq.parameters():
        p.requires_grad_(False)

    mask = make_mask().to(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(BASE.glob('*.jpg'))[:args.limit]
    report = {}
    for p in files:
        t0 = __import__('time').time()
        a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
        H, W = a.shape[:2]
        img0 = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
        base_s = float(cq(img0.clamp(1e-4, 1 - 1e-4)).mean().item())

        canvas = torch.zeros(1, 3, H, W, device=device)
        wsum = torch.zeros(1, 1, H, W, device=device)
        step = TILE - OVERLAP
        ys = list(range(0, max(H - TILE, 0) + 1, step))
        xs = list(range(0, max(W - TILE, 0) + 1, step))
        if (H - TILE) > ys[-1]:
            ys.append(H - TILE)
        if (W - TILE) > xs[-1]:
            xs.append(W - TILE)
        for y0 in ys:
            for x0 in xs:
                tile0 = img0[:, :, y0:y0 + TILE, x0:x0 + TILE].clone()
                x = tile0.clone().requires_grad_(True)
                opt = torch.optim.Adam([x], lr=args.lr)
                for _ in range(args.steps):
                    opt.zero_grad()
                    xc = x.clamp(0, 1)
                    score = cq(xc.clamp(1e-4, 1 - 1e-4)).mean()
                    (-score).backward()
                    opt.step()
                    with torch.no_grad():
                        x.clamp_(0, 1)
                        d = x - tile0
                        if d.abs().max() > args.budget:
                            x.copy_(tile0 + d.clamp(-args.budget, args.budget))
                with torch.no_grad():
                    upd = (x.detach() - tile0) * mask + tile0
                    canvas[:, :, y0:y0 + TILE, x0:x0 + TILE] += upd * mask
                    wsum[:, :, y0:y0 + TILE, x0:x0 + TILE] += mask
        final = canvas / wsum
        final_s = float(cq(final.clamp(1e-4, 1 - 1e-4)).mean().item())
        arr = (final[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out / p.name, quality=98)
        report[p.stem] = {'base': round(base_s, 5), 'final': round(final_s, 5),
                          'gain': round(final_s - base_s, 5),
                          'secs': round(__import__('time').time() - t0, 1)}
        print(f'{p.stem}: base={base_s:.4f} final={final_s:.4f} gain={final_s - base_s:+.4f} '
              f'({report[p.stem]["secs"]}s)', flush=True)
        del x, canvas, wsum, final
        torch.cuda.empty_cache()

    gains = [v['gain'] for v in report.values()]
    secs = [v['secs'] for v in report.values()]
    print(f'\n[summary] mean gain={np.mean(gains):+.4f}, avg {np.mean(secs):.0f}s/img', flush=True)
    json.dump(report, open(r'D:/tmp_timing/clip_ascent_report.json', 'w'), indent=1)
    print('[ok]', flush=True)


if __name__ == '__main__':
    main()
