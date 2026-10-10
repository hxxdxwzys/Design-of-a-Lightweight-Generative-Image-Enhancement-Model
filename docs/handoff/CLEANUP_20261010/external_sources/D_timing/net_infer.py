# -*- coding: utf-8 -*-
"""NAFNet 4K 分块推理 + 验证 (val: 3 对 / test: 100 张)
用法:
  python net_infer.py --ckpt D:/models/metric_net/net_s5000.pt --mode val
  python net_infer.py --ckpt ... --mode test --out D:/tmp_timing/cnn_out
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:\tmp_timing')
from train_metric_net import NAFNet  # noqa: E402

VAL_DIR = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\验证集')
TEST_DIR = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集')

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
    """img_pil: PIL RGB; 返回 [1,3,H,W] [0,1] fp32 cuda"""
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
                out = net(p)  # fp32 推理 (fp16 LayerNorm 在平坦 patch 会 NaN)
                out = out.clamp(0, 1)
                acc[:, :, y0:y0 + PATCH, x0:x0 + PATCH] += out * mask
                wsum[:, :, y0:y0 + PATCH, x0:x0 + PATCH] += mask
    return acc / wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--mode', choices=['val', 'test'], required=True)
    ap.add_argument('--out', default=r'D:\tmp_timing\cnn_val_out')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    device = 'cuda'
    net = NAFNet().to(device).eval()  # fp32 推理
    ck = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ck['net'])
    print(f'[load] {args.ckpt} step={ck.get("step")}', flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'val':
        for case in (1, 2, 3):
            lq_p = VAL_DIR / f'case{case}_lq.jpg'
            img = Image.open(lq_p).convert('RGB')
            out = tiled_infer(net, img, device)
            arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / f'case{case}_pred.jpg', quality=98)
            print(f'[val] case{case} rendered', flush=True)
        # 直接 eval lpips + clipiqa
        import pyiqa
        lpips_m = pyiqa.create_metric('lpips', device=device)
        cq_m = pyiqa.create_metric('clipiqa', device=device)
        tag = args.tag or args.ckpt.replace('\\', '/').split('/')[-1]
        tot = {'lpips': [], 'clipiqa': []}
        for case in (1, 2, 3):
            gt_p = VAL_DIR / f'case{case}_gt.png'
            if not gt_p.exists():
                gt_p = VAL_DIR / f'case{case}_gt.jpg'
            def load(p):
                a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
                return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = load(out_dir / f'case{case}_pred.jpg').clamp(1e-4, 1 - 1e-4)
            gt = load(gt_p).clamp(1e-4, 1 - 1e-4)
            lp = float(lpips_m(pred, gt).mean().item())
            cq = float(cq_m(pred.clamp(1e-4, 1 - 1e-4)).mean().item())
            tot['lpips'].append(lp)
            tot['clipiqa'].append(cq)
            print(f'  case{case}: lpips={lp:.4f} clipiqa={cq:.4f}', flush=True)
        import json
        res = {k: sum(v) / len(v) for k, v in tot.items()}
        print(f'=== {tag}: lpips={res["lpips"]:.4f} clipiqa={res["clipiqa"]:.4f}', flush=True)
        with open(out_dir / 'val_result.json', 'w') as f:
            json.dump({'tag': tag, **res}, f)
        # 公式得分 (质量分量, 相对 LQ 基线)
        q = 13.5 + 32 * (0.4262 - res['lpips']) + 40 * (res['clipiqa'] - 0.2509)
        print(f'=== 公式质量分量: {q:.1f} (HYPIR=23.9)', flush=True)
    else:
        for p in sorted(TEST_DIR.glob('*.jpg')):
            img = Image.open(p).convert('RGB')
            out = tiled_infer(net, img, device)
            arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / f'{p.stem}.jpg', quality=98)
            print(f'[test] {p.stem}', flush=True)


if __name__ == '__main__':
    main()
