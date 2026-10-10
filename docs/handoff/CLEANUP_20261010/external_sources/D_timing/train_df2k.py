# -*- coding: utf-8 -*-
"""DF2K 重训 (质量侧决胜): NAFNet(可调宽度) + LPIPS 主导损失

相对 train_metric_net.py 的改进:
- 数据: 多目录 (--data 逗号分隔, 如 DIV2K + Flickr2K = DF2K 3450 张)
- 网络: --width 可调 (32=36M / 48=81M / 64=144M)
- 保存: 每 --save_every 步存 weights-only (net_sN.pt, 小) + 每 5000 步存 full (含 opt, 供 resume)
- 损失/退化/lr/patch: 沿用 s6k 已验证配方 (LPIPS 3.0 + L1 0.05, 无 clipiqa, 256 patch)
"""
import argparse
import io
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter

import pyiqa


# ---------------- 退化 (s6k 校准范围, 不变) ----------------

def _jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_pil(img):
    order = 'j_b_n' if random.random() < 0.8 else 'b_n_j'
    s = random.uniform(1.2, 4.2)
    q = random.randint(50, 96)
    n = random.uniform(0.0, 1.5)
    if order == 'j_b_n':
        im = _jpeg(img, q)
        im = im.filter(ImageFilter.GaussianBlur(radius=s))
    else:
        im = img.filter(ImageFilter.GaussianBlur(radius=s))
        im = _jpeg(im, q)
    if n > 0:
        arr = np.asarray(im, dtype=np.float32)
        arr = np.clip(arr + np.random.randn(*arr.shape).astype(np.float32) * n, 0, 255)
        im = Image.fromarray(arr.astype(np.uint8))
    return im


# ---------------- NAFNet (宽度可调) ----------------

class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(c))
        self.beta = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        m = x.mean((1, 2), keepdim=True)
        v = x.var((1, 2), unbiased=False, keepdim=True)
        return (x - m) / (v + self.eps).sqrt() * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class SCA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        v = x.mean((2, 3), keepdim=True)
        v = self.conv(v)
        return x * v


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, c * 2, 3, padding=1, bias=False),
            SimpleGate(),
            SCA(c),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x) + x


class NAFNet(nn.Module):
    def __init__(self, width=32, enc=(2, 2, 4, 8), mid=(10,), dec=(8, 4, 2, 2)):
        super().__init__()
        self.width = width
        self.intro = nn.Conv2d(3, width, 3, padding=1, bias=False)
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = width
        for i, n in enumerate(enc):
            self.enc.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            if i < len(enc) - 1:
                self.downs.append(nn.Conv2d(c, c * 2, 2, stride=2, bias=False))
                c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in mid])
        self.dec = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, n in enumerate(dec):
            if i > 0:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(c, c // 2, 3, padding=1, bias=False)))
                c //= 2
            self.dec.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.tail = nn.Sequential(LayerNorm2d(c), nn.Conv2d(c, 3, 3, padding=1, bias=False))

    def forward(self, x):
        inp = x
        x = self.intro(x)
        skips = []
        h = x
        for i, blk in enumerate(self.enc):
            h = blk(h)
            if i < len(self.downs):
                skips.append(h)
                h = self.downs[i](h)
        h = self.mid(h)
        for i, blk in enumerate(self.dec):
            if i > 0:
                h = self.ups[i - 1](h)
                h = h + skips.pop()
            h = blk(h)
        return self.tail(h) + inp


# ---------------- 训练 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:/datasets/div2k/pngtxt_ori', help='逗号分隔多个 HQ 图目录')
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--patch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--w_l1', type=float, default=0.05)
    ap.add_argument('--w_lpips', type=float, default=3.0)
    ap.add_argument('--out', default=r'D:/models/metric_net_f2k')
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dirs = [Path(d.strip()) for d in args.data.split(',') if d.strip()]
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    imgs = []
    for d in data_dirs:
        for e in exts:
            imgs += sorted(d.glob(e))
    # 去重 (防 DIV2K png+jpg 同源重复)
    seen = set()
    uniq = []
    for p in imgs:
        k = p.stem
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    imgs = uniq
    print(f'[data] {len(imgs)} HQ images from {len(data_dirs)} dir(s)', flush=True)

    net = NAFNet(width=args.width).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'[net] width={args.width} params: {n_params/1e6:.2f}M', flush=True)

    lpips_metric = pyiqa.create_metric('lpips', device=device)
    lpips_metric.as_loss = True
    for p in lpips_metric.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck['net'])
        opt.load_state_dict(ck['opt'])
        start = ck['step']
        for _ in range(start):
            sched.step()
        print(f'[resume] from step {start}', flush=True)

    def rand_crop(im):
        w, h = im.size
        if w < args.patch or h < args.patch:
            im = im.resize((max(w, args.patch), max(h, args.patch)), Image.BICUBIC)
            w, h = im.size
        x0 = random.randint(0, w - args.patch)
        y0 = random.randint(0, h - args.patch)
        return im.crop((x0, y0, x0 + args.patch, y0 + args.patch))

    def aug(pil):
        if random.random() < 0.5:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            pil = pil.transpose(Image.ROTATE_90)
        return pil

    def pil2t(pil):
        arr = np.asarray(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    POOL_SIZE = 2400 if args.patch < 384 else 800

    def build_pool():
        pool = []
        for _ in range(POOL_SIZE):
            im = Image.open(random.choice(imgs)).convert('RGB')
            h = aug(rand_crop(im))
            l = degrade_pil(h)
            pool.append((pil2t(l), pil2t(h)))
        return pool

    pool = build_pool()
    print(f'[data] pool built: {len(pool)} pairs', flush=True)

    def make_batch():
        lq, hq = [], []
        for l, h in random.sample(pool, args.batch):
            lq.append(l)
            hq.append(h)
        return torch.stack(lq).to(device), torch.stack(hq).to(device)

    print('[train] start (全 fp32)', flush=True)
    t0 = time.time()
    for step in range(start, args.steps):
        lq, hq = make_batch()
        pred = net(lq)
        p = pred.clamp(1e-4, 1 - 1e-4)
        g = hq.clamp(1e-4, 1 - 1e-4)
        loss_l1 = F.l1_loss(p, g)
        loss_lp = lpips_metric(p, g).mean()
        loss = args.w_l1 * loss_l1 + args.w_lpips * loss_lp
        if not torch.isfinite(loss):
            print(f'[warn] step {step} loss NaN, skip', flush=True)
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            el = (time.time() - t0) / (step - start + 1)
            print(f'[step {step}] loss={loss.item():.4f} l1={loss_l1.item():.4f} '
                  f'lpips={loss_lp.item():.4f} lr={sched.get_last_lr()[0]:.2e} {el:.2f}s/step', flush=True)
        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            torch.save({'net': net.state_dict(), 'step': step + 1}, f'{args.out}/net_s{step+1}.pt')
            print(f'[save] net_s{step+1}.pt (weights-only)', flush=True)
        if (step + 1) % 5000 == 0:
            torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'step': step + 1},
                       f'{args.out}/full_s{step+1}.pt')
            print(f'[save] full_s{step+1}.pt', flush=True)
        if step > 0 and step % 3000 == 0:
            pool = build_pool()
            print('[data] pool refreshed', flush=True)

    print('[done]', flush=True)


if __name__ == '__main__':
    main()
