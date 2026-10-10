# -*- coding: utf-8 -*-
"""train_v2.py: 测试集退化匹配重训 (2026-08-14 deepseek冲刺4)

背景: 测试集实测退化 = 轻 JPEG (89% 图 q>80) + 轻模糊 (中位 sigma~0.5) + 中重噪声 (中位 n~1.06, 23%>2.4),
      而旧训练采样器 = 重模糊(1.2-4.2)+轻噪声(0-1.5) -> 错位是"更多数据无效"的根因。

与 train_df2k.py 的差异:
- 经验分布退化采样器: 从 degradation_analysis.json + jpeg_blockiness.json 读 100 张测试图的
  (sigma,n,q) 三元组, 每次随机选一张的测量值 + 微扰 -> 训练分布直接匹配测试集
- EMA 权重滑动平均 (decay 0.999), 每 --save_every 步存 ema_sN.pt
- 留出验证集: 40 张 HQ (固定 seed 排除) 用同一经验采样器退化一次 -> 每 --holdout_every 步
  tiled 推理 + lpips/clipiqa (原生分辨率); 官方 3-val (4K) 同步评测 (与旧口径可比)
- 其余 (NAFNet/LPIPS3.0+L1 0.05/cosine/池化) 沿用 s6k 配方

用法:
  D:/Python/python.exe train_v2.py --out D:/models/metric_net_v2 --steps 16000
  D:/Python/python.exe train_v2.py --out D:/models/metric_net_v2 --width 48 --steps 16000 --batch 4
"""
import argparse
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter

import pyiqa

DEG_JSON = r'D:/tmp_timing/degradation_analysis.json'
BLK_JSON = r'D:/tmp_timing/jpeg_blockiness.json'
VAL_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
PATCH_INFER = 512
OVERLAP = 128


def _jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


class FixedProfile:
    """固定范围退化 (v2b 修正: 与 s6k 配方对齐但噪声范围上扩)"""

    def __init__(self, s_min=0.8, s_max=3.5, n_min=0.0, n_max=2.0, q_min=55, q_max=95):
        self.ranges = (s_min, s_max, n_min, n_max, q_min, q_max)
        print(f'[profile] fixed: sigma[{s_min},{s_max}] noise[{n_min},{n_max}] jpeg[{q_min},{q_max}]', flush=True)

    def params(self):
        s_min, s_max, n_min, n_max, q_min, q_max = self.ranges
        return random.uniform(s_min, s_max), random.uniform(n_min, n_max), random.randint(q_min, q_max)

    def degrade(self, img):
        s, n, q = self.params()
        order = 'j_b_n' if random.random() < 0.8 else 'b_n_j'
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


class EmpProfile:
    """经验退化分布: 从测试集实测 (sigma, n, q) 三元组 bootstrap"""

    def __init__(self, deg_json, blk_json):
        self.sigmas, self.noises, self.qs = [], [], []
        try:
            deg = json.load(open(deg_json))
            blk = json.load(open(blk_json))
            by_name = {r['name']: r for r in blk['per_image'] if r['tag'] == 'test'}
            for r in deg['per_image']:
                if r['tag'] != 'test':
                    continue
                b = by_name.get(r['name'], {})
                self.sigmas.append(float(r.get('est_sigma', 0.8)))
                self.noises.append(float(r.get('est_noise_n', 1.0)))
                self.qs.append(float(b.get('est_jpeg_q', 88)))
            print(f'[profile] 经验分布加载成功: {len(self.sigmas)} 个测试图三元组', flush=True)
        except Exception as e:
            print(f'[profile] 经验分布加载失败({e}), 用兜底范围', flush=True)

    def params(self):
        if self.sigmas:
            i = random.randrange(len(self.sigmas))
            s = max(0.2, self.sigmas[i] + random.uniform(-0.15, 0.25))
            n = max(0.0, self.noises[i] + random.uniform(-0.1, 0.15))
            q = int(round(min(96, max(30, self.qs[i] + random.randint(-2, 2)))))
            return s, n, q
        return random.uniform(0.3, 1.8), random.uniform(0.3, 2.5), random.randint(60, 95)

    def degrade(self, img):
        s, n, q = self.params()
        order = 'j_b_n' if random.random() < 0.8 else 'b_n_j'
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


# ---------------- NAFNet (与 train_df2k 相同) ----------------

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
        self.c = c
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


# ---------------- 推理/评测工具 ----------------

def make_mask():
    m = torch.ones(PATCH_INFER, PATCH_INFER)
    ramp = OVERLAP // 2
    for i in range(ramp):
        w = (i + 1) / ramp
        m[i, :] = w
        m[PATCH_INFER - 1 - i, :] = w
        m[:, i] = torch.minimum(m[:, i], torch.full((PATCH_INFER,), w))
        m[:, PATCH_INFER - 1 - i] = torch.minimum(m[:, PATCH_INFER - 1 - i], torch.full((PATCH_INFER,), w))
    return m


def tiled_infer(net, img_pil, device, mask):
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    acc = torch.zeros(1, 3, H, W, device=device)
    wsum = torch.zeros(1, 1, H, W, device=device)
    step = PATCH_INFER - OVERLAP
    ys = list(range(0, max(H - PATCH_INFER, 0) + 1, step))
    xs = list(range(0, max(W - PATCH_INFER, 0) + 1, step))
    if (H - PATCH_INFER) > ys[-1]:
        ys.append(H - PATCH_INFER)
    if (W - PATCH_INFER) > xs[-1]:
        xs.append(W - PATCH_INFER)
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                p = x[:, :, y0:y0 + PATCH_INFER, x0:x0 + PATCH_INFER]
                out = net(p).clamp(0, 1)
                acc[:, :, y0:y0 + PATCH_INFER, x0:x0 + PATCH_INFER] += out * mask
                wsum[:, :, y0:y0 + PATCH_INFER, x0:x0 + PATCH_INFER] += mask
    return acc / wsum


def load_t(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:/datasets/div2k/pngtxt_ori,D:/datasets/flickr2k')
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--steps', type=int, default=16000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--accum', type=int, default=1, help='梯度累积步数 (有效 batch = batch*accum)')
    ap.add_argument('--patch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--w_l1', type=float, default=0.05)
    ap.add_argument('--w_lpips', type=float, default=3.0)
    ap.add_argument('--out', default=r'D:/models/metric_net_v2')
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--holdout', default=r'D:/datasets/holdout_v2')
    ap.add_argument('--holdout_every', type=int, default=2000)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--sampler', choices=['emp', 'fixed'], default='emp')
    ap.add_argument('--sigma-min', type=float, default=0.8)
    ap.add_argument('--sigma-max', type=float, default=3.5)
    ap.add_argument('--noise-min', type=float, default=0.0)
    ap.add_argument('--noise-max', type=float, default=2.0)
    ap.add_argument('--jpeg-min', type=int, default=55)
    ap.add_argument('--jpeg-max', type=int, default=95)
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
    seen, uniq = set(), []
    for p in imgs:
        if p.stem not in seen:
            seen.add(p.stem)
            uniq.append(p)
    imgs = uniq
    print(f'[data] {len(imgs)} HQ images', flush=True)

    if args.sampler == 'fixed':
        profile = FixedProfile(args.sigma_min, args.sigma_max, args.noise_min, args.noise_max,
                               args.jpeg_min, args.jpeg_max)
    else:
        profile = EmpProfile(DEG_JSON, BLK_JSON)

    # 留出验证集: 40 张, 固定 seed, 与训练数据互斥
    hd = Path(args.holdout)
    rng_h = random.Random(1234)
    hold_imgs = rng_h.sample(imgs, min(40, len(imgs)))
    hold_imgs = set(hold_imgs)
    train_imgs = [p for p in imgs if p not in hold_imgs]
    print(f'[holdout] {len(hold_imgs)} 张留出, {len(train_imgs)} 张训练', flush=True)
    hd.mkdir(parents=True, exist_ok=True)
    lq_dir, gt_dir = hd / 'lq', hd / 'gt'
    lq_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    if not (hd / 'manifest.json').exists():
        manifest = []
        for p in sorted(hold_imgs):
            im = Image.open(p).convert('RGB')
            lq = profile.degrade(im)
            name = p.stem
            lq.save(lq_dir / f'{name}_lq.png')
            im.save(gt_dir / f'{name}_gt.png')
            manifest.append(name)
            print(f'[holdout] built {name}', flush=True)
        json.dump(manifest, open(hd / 'manifest.json', 'w'))
        print(f'[holdout] 40 对已生成', flush=True)
    else:
        manifest = json.load(open(hd / 'manifest.json'))
        print(f'[holdout] 复用已有 {len(manifest)} 对', flush=True)

    net = NAFNet(width=args.width).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'[net] width={args.width} params: {n_params/1e6:.2f}M', flush=True)

    lpips_metric = pyiqa.create_metric('lpips', device=device)
    lpips_metric.as_loss = True
    for p in lpips_metric.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps // args.accum), eta_min=1e-6)
    start = 0
    ema = None
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck['net'])
        opt.load_state_dict(ck['opt'])
        start = ck['step']
        for _ in range(start):
            sched.step()
        ema = ck.get('ema')
        print(f'[resume] from step {start}', flush=True)
    if ema is None:
        ema = {k: v.detach().clone() for k, v in net.state_dict().items()}

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
            im = Image.open(random.choice(train_imgs)).convert('RGB')
            h = aug(rand_crop(im))
            l = profile.degrade(h)
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

    # ---- 验证函数 ----
    mask = make_mask().to(device)
    lpips_eval = pyiqa.create_metric('lpips', device=device)
    cq_eval = pyiqa.create_metric('clipiqa', device=device)

    def run_eval(tag):
        net.eval()
        with torch.no_grad():
            # 留出集
            lps, cqs = [], []
            for name in manifest:
                lq = Image.open(lq_dir / f'{name}_lq.png').convert('RGB')
                gt = Image.open(gt_dir / f'{name}_gt.png').convert('RGB')
                out = tiled_infer(net, lq, device, mask)
                pred = out.clamp(1e-4, 1 - 1e-4)
                g = load_t(gt_dir / f'{name}_gt.png').to(device).clamp(1e-4, 1 - 1e-4)
                lps.append(float(lpips_eval(pred, g).mean().item()))
                cqs.append(float(cq_eval(pred).mean().item()))
            hl, hc = float(np.mean(lps)), float(np.mean(cqs))
            print(f'[eval:{tag}] holdout({len(manifest)}): lpips={hl:.4f} clipiqa={hc:.4f}', flush=True)
            # 官方 3-val (4K, 与旧口径可比)
            vl, vc = [], []
            for case in (1, 2, 3):
                lq_p = VAL_DIR / f'case{case}_lq.jpg'
                gt_p = VAL_DIR / f'case{case}_gt.png'
                if not gt_p.exists():
                    gt_p = VAL_DIR / f'case{case}_gt.jpg'
                out = tiled_infer(net, Image.open(lq_p).convert('RGB'), device, mask)
                pred = out.clamp(1e-4, 1 - 1e-4)
                g = load_t(gt_p).to(device).clamp(1e-4, 1 - 1e-4)
                vl.append(float(lpips_eval(pred, g).mean().item()))
                vc.append(float(cq_eval(pred).mean().item()))
            print(f'[eval:{tag}] val3(4K): lpips={np.mean(vl):.4f} clipiqa={np.mean(vc):.4f}', flush=True)
            return {'holdout_lpips': hl, 'holdout_clipiqa': hc,
                    'val3_lpips': float(np.mean(vl)), 'val3_clipiqa': float(np.mean(vc))}
        # net.train()

    eval_log = []
    print('[train] start (全 fp32)', flush=True)
    t0 = time.time()
    for step in range(start, args.steps):
        net.train()
        lq, hq = make_batch()
        pred = net(lq)
        p = pred.clamp(1e-4, 1 - 1e-4)
        g = hq.clamp(1e-4, 1 - 1e-4)
        loss_l1 = F.l1_loss(p, g)
        loss_lp = lpips_metric(p, g).mean()
        loss_full = args.w_l1 * loss_l1 + args.w_lpips * loss_lp
        loss = loss_full / args.accum
        if not torch.isfinite(loss):
            print(f'[warn] step {step} loss NaN, skip', flush=True)
            opt.zero_grad()
            continue
        loss.backward()
        if (step + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        with torch.no_grad():
            for k, v in net.state_dict().items():
                ema[k].mul_(args.ema_decay).add_(v.detach(), alpha=1 - args.ema_decay)
        if step % 100 == 0:
            el = (time.time() - t0) / (step - start + 1)
            print(f'[step {step}] loss={loss_full.item():.4f} l1={loss_l1.item():.4f} '
                  f'lpips={loss_lp.item():.4f} lr={sched.get_last_lr()[0]:.2e} {el:.2f}s/step', flush=True)
        if (step + 1) % args.save_every == 0:
            ck = {'net': net.state_dict(), 'opt': opt.state_dict(), 'step': step + 1}
            torch.save(ck, f'{args.out}/net_s{step+1}.pt')
            torch.save({'net': ema}, f'{args.out}/ema_s{step+1}.pt')
            if (step + 1) % 5000 == 0:
                ck['ema'] = ema
                torch.save(ck, f'{args.out}/full_s{step+1}.pt')
            print(f'[save] net_s{step+1}.pt + ema_s{step+1}.pt', flush=True)
        if (step + 1) % args.holdout_every == 0:
            r = run_eval(step + 1)
            eval_log.append({'step': step + 1, **r})
            with open(f'{args.out}/eval_log.json', 'w') as f:
                json.dump(eval_log, f, indent=1)
        if step > 0 and step % 3000 == 0:
            pool = build_pool()
            print('[data] pool refreshed', flush=True)

    ck = {'net': net.state_dict(), 'opt': opt.state_dict(), 'step': args.steps, 'ema': ema}
    torch.save(ck, f'{args.out}/full_final.pt')
    torch.save({'net': ema}, f'{args.out}/ema_final.pt')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
