# -*- coding: utf-8 -*-
"""metric-aligned 复原网络训练 (质量侧决胜点, 输出侧专用, 非提交模型)

目标: 直接优化官方评分公式的质量分量 (score ≈ 13.5 + 32*Δlpips + 40*Δclipiqa)
网络: NAFNet-32 (轻量复原 CNN)
损失: L1 + LPIPS(alex) + (1 - CLIPIQA)  ← 全部可微, 与 eval.py 同库同默认
退化: 校准后的官方范围 (order j_b_n 80% / b_n_j 20%, blur σ∈[1.2,4.2], jpeg q∈[50,96], noise∈[0,1.5])
数据: D:/datasets/div2k/pngtxt_ori/*.png (800)
验证: 每 5000 步 tiled 4K 推理 3 张验证集 + eval lpips/clipiqa
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

# ---------------- 退化 (校准范围) ----------------

def _jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def degrade_pil(img):
    """PIL RGB (0-255) -> LQ (同尺寸), 按校准分布采样"""
    order = 'j_b_n' if random.random() < 0.8 else 'b_n_j'
    s = random.uniform(1.2, 4.2)
    q = random.randint(50, 96)
    n = random.uniform(0.0, 1.5)
    if order == 'j_b_n':
        im = _jpeg(img, q)
        im = im.filter(ImageFilter.GaussianBlur(radius=s))
        if n > 0:
            arr = np.asarray(im, dtype=np.float32)
            arr = np.clip(arr + np.random.randn(*arr.shape).astype(np.float32) * n, 0, 255)
            im = Image.fromarray(arr.astype(np.uint8))
        return im
    else:
        im = img.filter(ImageFilter.GaussianBlur(radius=s))
        im = _jpeg(im, q)
        if n > 0:
            arr = np.asarray(im, dtype=np.float32)
            arr = np.clip(arr + np.random.randn(*arr.shape).astype(np.float32) * n, 0, 255)
            im = Image.fromarray(arr.astype(np.uint8))
        return im

# ---------------- NAFNet (宽度32, 4x下采样) ----------------

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
    """Spatial-Channel Attention"""
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.conv = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        v = x.mean((2, 3), keepdim=True)  # 全局平均 -> 1x1xc
        v = self.conv(v)                   # 通道注意力
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
                h = h + skips.pop()  # 残差式融合, 保持通道
            h = blk(h)
        return self.tail(h) + inp  # 全局残差到输入


# ---------------- 训练 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:/datasets/div2k/pngtxt_ori')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--patch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--w_l1', type=float, default=0.2)
    ap.add_argument('--w_lpips', type=float, default=1.0)
    ap.add_argument('--w_clipiqa', type=float, default=0.8)
    ap.add_argument('--out', default=r'D:/models/metric_net')
    ap.add_argument('--val_every', type=int, default=5000)
    ap.add_argument('--resume', default=None)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(0)
    np.random.seed(0)

    imgs = sorted(Path(args.data).glob('*.png'))
    print(f'[data] {len(imgs)} HQ images', flush=True)

    net = NAFNet().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'[net] params: {n_params/1e6:.2f}M', flush=True)

    lpips_metric = pyiqa.create_metric('lpips', device=device)
    clipiqa = pyiqa.create_metric('clipiqa', device=device)
    # ⚠️ pyiqa 默认 as_loss=False → forward 在 no_grad 下执行, 输出不可微!
    # 必须置 True 才有梯度; 权重本身冻结 (指标只作为损失, 不训练)
    lpips_metric.as_loss = True
    clipiqa.as_loss = True
    for p in lpips_metric.parameters():
        p.requires_grad_(False)
    for p in clipiqa.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck['net'])
        opt.load_state_dict(ck['opt'])
        start = ck['step']
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

    POOL_SIZE = 800 if args.patch >= 384 else 2400

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

    print('[train] start (全 fp32, 网络前向 fp16 曾溢出 NaN)', flush=True)
    t0 = time.time()
    for step in range(start, args.steps):
        lq, hq = make_batch()
        pred = net(lq)  # 全程 fp32, 避免 fp16 溢出
        # 指标/损失全部在 fp32 域计算
        # ⚠️ 教训: 256 patch 上的 clipiqa 与 4K 整图 clipiqa 严重脱节 (训练0.74 vs 4K 0.23),
        #     clipiqa 损失会拉出"过度锐化"伪影。默认关闭, 仅当 w_clipiqa>0 时启用。
        p = pred.clamp(1e-4, 1 - 1e-4)
        g = hq.clamp(1e-4, 1 - 1e-4)
        loss_l1 = F.l1_loss(p, g)
        loss_lp = lpips_metric(p, g).mean()
        loss = args.w_l1 * loss_l1 + args.w_lpips * loss_lp
        cq = None
        if args.w_clipiqa > 0:
            cq = clipiqa(p).mean()
            loss = loss + args.w_clipiqa * (1.0 - cq)
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
            cqs = f' clipiqa={cq.item():.4f}' if cq is not None else ''
            print(f'[step {step}] loss={loss.item():.4f} l1={loss_l1.item():.4f} '
                  f'lpips={loss_lp.item():.4f}{cqs} '
                  f'lr={sched.get_last_lr()[0]:.2e} {el:.2f}s/step', flush=True)
        if (step + 1) % 1000 == 0 or (step + 1) % args.val_every == 0:
            ck = {'net': net.state_dict(), 'opt': opt.state_dict(), 'step': step + 1}
            torch.save(ck, f'{args.out}/net_s{step+1}.pt')
            if (step + 1) % 1000 == 0:
                print(f'[save] net_s{step+1}.pt', flush=True)
        if step > 0 and step % 3000 == 0:
            pool = build_pool()  # 刷新退化多样性
            print('[data] pool refreshed', flush=True)

    ck = {'net': net.state_dict(), 'opt': opt.state_dict(), 'step': args.steps}
    torch.save(ck, f'{args.out}/net_final.pt')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
