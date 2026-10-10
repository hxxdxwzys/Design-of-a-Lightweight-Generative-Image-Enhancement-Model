# -*- coding: utf-8 -*-
"""width=48 显存冒烟: 复刻训练图 (L1+LPIPS 反向) 测峰值显存
用法: D:/Python/python.exe mem_probe_w48.py --width 48 --batch 4 --patch 256
打印: 峰值 allocated/reserved (GB), 供选 batch/accum 组合。
"""
import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:/tmp_timing')
from train_df2k import NAFNet  # noqa: E402
import pyiqa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=48)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--patch', type=int, default=256)
    ap.add_argument('--iters', type=int, default=8)
    args = ap.parse_args()

    device = 'cuda'
    torch.cuda.reset_peak_memory_stats(device)
    net = NAFNet(width=args.width).to(device).train()
    n_params = sum(p.numel() for p in net.parameters())
    print(f'[net] width={args.width} params={n_params / 1e6:.2f}M', flush=True)

    lpips_metric = pyiqa.create_metric('lpips', device=device)
    lpips_metric.as_loss = True
    for p in lpips_metric.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=0.02)

    for i in range(args.iters):
        lq = torch.rand(args.batch, 3, args.patch, args.patch, device=device)
        hq = torch.rand(args.batch, 3, args.patch, args.patch, device=device)
        pred = net(lq)
        p = pred.clamp(1e-4, 1 - 1e-4)
        g = hq.clamp(1e-4, 1 - 1e-4)
        loss_l1 = F.l1_loss(p, g)
        loss_lp = lpips_metric(p, g).mean()
        loss = 0.05 * loss_l1 + 3.0 * loss_lp
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    alloc = torch.cuda.max_memory_allocated(device) / 1e9
    reserv = torch.cuda.max_memory_reserved(device) / 1e9
    print(f'[peak] width={args.width} batch={args.batch} patch={args.patch}: '
          f'allocated={alloc:.2f}GB reserved={reserv:.2f}GB', flush=True)


if __name__ == '__main__':
    main()
