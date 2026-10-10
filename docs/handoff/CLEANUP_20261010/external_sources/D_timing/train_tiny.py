# -*- coding: utf-8 -*-
"""微模型 v1 训练 (2026-09-13 夜) —— 速度轴已证实的直接执行

事实基础(今天实测):
  * 同图只改时延: M24(0.780x) 43.6951 -> avg2(0.382x) 36.6414 -> Slow8(0.098x) 28.7538
    => 4.98~6.84 分/翻倍, 且越靠近 1x 越陡 (凹曲线)
  * 细节轴证伪: ProbeDetail(lapvar x2.27) 43.0699 (-0.63)
  => 目标: 造一个"真能在 10~40x 跑、图像不烂"的 1 步扩散模型

模型: diffusers UNet2DModel, 像素空间, 1 步 eps 参数化 (alpha 取 t=500), 与旧探针同族
数据: DIV2K + Flickr2K HQ, 在线按【测试集实测退化分布】合成 LQ
      轻模糊 sigma~|N(0,0.7)| (中位~0.5) / 噪声 n~lognormal(中位~1.05) / JPEG q~U(80,100)
损失: L1(x0) + 0.2 * LPIPS(x0)  (感知项让输出不发糊)
用法: python train_tiny.py [--res 512] [--steps 30000] [--bs 8] [--ch 32,32,32,64]
"""
from __future__ import annotations
import argparse, io, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

os.environ.setdefault('HF_HUB_OFFLINE', '1')
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)

HQ_DIRS = [Path(r'D:/datasets/div2k/pngtxt_ori'), Path(r'D:/datasets/flickr2k')]
OUT = Path(r'D:/models/tiny_v1')
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / 'train.log'


def log(m):
    s = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


def list_hq():
    files = []
    for d in HQ_DIRS:
        files += sorted(d.glob('*.png'))
        files += sorted(d.glob('*.jpg'))
    return files


from concurrent.futures import ThreadPoolExecutor
POOL = ThreadPoolExecutor(max_workers=10)   # PIL 的 jpeg/blur 会释放 GIL -> 线程池够用
PATCH_FILES = []


DEG_SMALL = 0        # >1 时: 先把 HQ 降到目标域再退化 (快 3-4 倍, 分布近似)
DEG_DOWN = 1


def degrade_small(arr_u8, k):
    """在目标域退化: 强度按 k=1/down 缩放 (对应"先退化再降采样"的等效分布)"""
    im = Image.fromarray(arr_u8)
    q = int(np.random.uniform(88, 100))
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=q, subsampling=int(np.random.choice([0, 2])))
    buf.seek(0)
    im = Image.open(buf).convert('RGB')
    sig = min(abs(float(np.random.normal(0, 0.7))) * k, 1.0)
    if sig > 0.03:
        im = im.filter(ImageFilter.GaussianBlur(radius=sig))
    a = np.asarray(im, dtype=np.float32)
    n = float(np.clip(math.exp(float(np.random.normal(math.log(1.05), 0.6))), 0.15, 3.5)) * k
    a = np.clip(a + np.random.standard_normal(a.shape).astype(np.float32) * n, 0, 255)
    return a.astype(np.uint8)


DISTILL_LQ = []
DISTILL_TGT = []


def load_pair(_i):
    """取一个 (HQ目标, LQ输入) 对; 蒸馏模式直接从磁盘读 (256 域)"""
    if DISTILL_LQ:
        for _ in range(6):          # 防御: 跳过 lq/tgt 尺寸不一致的脏数据
            k = np.random.randint(len(DISTILL_LQ))
            tgt = np.asarray(Image.open(DISTILL_TGT[k]).convert('RGB'), dtype=np.uint8)
            lq = np.asarray(Image.open(DISTILL_LQ[k]).convert('RGB'), dtype=np.uint8)
            if lq.shape == tgt.shape:
                return tgt, lq
        raise RuntimeError('distill 数据里找不到尺寸一致的一对')
    p = PATCH_FILES[np.random.randint(len(PATCH_FILES))]
    hq = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
    if DEG_SMALL and DEG_DOWN > 1:
        s = 512 // DEG_DOWN
        hqs = np.asarray(Image.fromarray(hq).resize((s, s), Image.BICUBIC), dtype=np.uint8)
        return hqs, degrade_small(hqs, 1.0 / DEG_DOWN)
    return hq, degrade(hq)


def degrade(arr_u8):
    """已裁剪的 HQ uint8 (H,W,3) -> LQ uint8; 按测试集实测分布
    轻模糊 sigma~|N(0,0.7)| (中位~0.5) / 噪声 n~lognormal(中位~1.05) / JPEG q~U(80,100)
    """
    im = Image.fromarray(arr_u8)
    q = int(np.random.uniform(80, 100))
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=q, subsampling=int(np.random.choice([0, 2])))
    buf.seek(0)
    im = Image.open(buf).convert('RGB')
    sig = min(abs(float(np.random.normal(0, 0.7))), 2.5)
    if sig > 0.05:
        im = im.filter(ImageFilter.GaussianBlur(radius=sig))
    arr = np.asarray(im, dtype=np.float32)
    n = float(np.clip(math.exp(float(np.random.normal(math.log(1.05), 0.6))), 0.15, 3.5))
    arr = np.clip(arr + np.random.standard_normal(arr.shape).astype(np.float32) * n, 0, 255)
    return arr.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--down', type=int, default=1, help='2 = 训练在降采样域(256), 推理时输入也降 2 倍 -> 4x 提速')
    ap.add_argument('--deg_small', type=int, default=0, help='1 = 在目标域做退化(快 3-4x, 分布近似)')
    ap.add_argument('--distill_dir', type=str, default='', help='蒸馏模式: 该目录下 lq/ 与 tgt/ 成对 (256 域)')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--ch', type=str, default='32,32,32,64')
    ap.add_argument('--w_lpips', type=float, default=0.2)
    ap.add_argument('--tag', type=str, default='v1')
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    dev = 'cuda'
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    rng = random.Random(1234)
    ch = tuple(int(x) for x in a.ch.split(','))

    from diffusers import UNet2DModel, DDPMScheduler
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear',
                          beta_start=0.00085, beta_end=0.012)
    alpha = sched.alphas_cumprod[500].to(dev).float()
    sq_a, sq_1a = alpha.sqrt(), (1 - alpha).sqrt()

    unet = UNet2DModel(sample_size=a.res // a.down, in_channels=3, out_channels=3,
                       layers_per_block=1, block_out_channels=ch,
                       down_block_types=('DownBlock2D',) * len(ch),
                       up_block_types=('UpBlock2D',) * len(ch)).to(dev)
    nparam = sum(p.numel() for p in unet.parameters()) / 1e6
    log(f'[model] ch={ch} params={nparam:.2f}M res={a.res} down={a.down} bs={a.bs} steps={a.steps} tag={a.tag}')

    import lpips as lpips_mod
    lp = lpips_mod.LPIPS(net='alex', verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(unet.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.05)
    try:
        scaler = torch.amp.GradScaler('cuda')
    except Exception:
        scaler = torch.cuda.amp.GradScaler()

    global PATCH_FILES, DEG_SMALL, DEG_DOWN, DISTILL_LQ, DISTILL_TGT
    DEG_SMALL = 1 if (a.down > 1 and a.deg_small) else 0
    DEG_DOWN = a.down
    if a.distill_dir:
        DISTILL_LQ = sorted((Path(a.distill_dir) / 'lq').glob('*.jpg'))
        DISTILL_TGT = sorted((Path(a.distill_dir) / 'tgt').glob('*.jpg'))
        assert len(DISTILL_LQ) == len(DISTILL_TGT) > 0, 'distill 目录不匹配'
        # 蒸馏对存的是 512 域原图 -> 由 --down 决定是否降采样 (走标准 resize 分支)
    PATCH_FILES = sorted(Path(r'D:/datasets/tiny_patches').glob('*.jpg'))
    if not PATCH_FILES:
        raise SystemExit('缺少预切 patch 缓存: 先跑 prep_patches.py')
    log(f'[data] {len(PATCH_FILES)} 个 512 patch (预切 JPEG 缓存)')

    t0 = time.time()
    run = 0.0
    for step in range(1, a.steps + 1):
        # 从预切 JPEG 缓存取 patch (快), load+degrade 一起丢进线程池
        pairs = list(POOL.map(load_pair, range(a.bs)))
        hq = np.stack([q[0] for q in pairs])
        lq = np.stack([q[1] for q in pairs])
        t_hq = torch.from_numpy(hq).permute(0, 3, 1, 2).to(dev).float().div_(127.5).sub_(1)
        t_lq = torch.from_numpy(lq).permute(0, 3, 1, 2).to(dev).float().div_(127.5).sub_(1)
        if a.down > 1 and not DEG_SMALL:   # 训练在降采样域: 推理时输入同样降 down 倍
            sz = (a.res // a.down, a.res // a.down)
            t_hq = F.interpolate(t_hq, size=sz, mode='bicubic', align_corners=False)
            t_lq = F.interpolate(t_lq, size=sz, mode='bicubic', align_corners=False)

        eps_t = (t_lq - sq_a * t_hq) / sq_1a          # 目标噪声
        with torch.autocast('cuda', dtype=torch.float16):
            pred = unet(t_lq, torch.full((a.bs,), 500, dtype=torch.long, device=dev)).sample
            x0 = (t_lq - sq_1a * pred.float()) / sq_a
            loss_l1 = F.l1_loss(x0, t_hq)
            loss_lp = lp(x0.clamp(-1, 1), t_hq).mean() if a.w_lpips > 0 else torch.zeros((), device=dev)
            loss = loss_l1 + a.w_lpips * loss_lp
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()

        run = 0.9 * run + 0.1 * float(loss.item()) if step > 1 else float(loss.item())
        if step in (5, 20, 50) or step % 100 == 0:
            log(f'step {step}/{a.steps} loss={run:.4f} (l1={float(loss_l1):.4f} lp={float(loss_lp):.4f}) '
                f'lr={sch.get_last_lr()[0]:.2e} {(time.time()-t0)/step:.2f}s/step')
        if step % 2000 == 0 or step == a.steps:
            ck = OUT / f'{a.tag}_s{step}.pt'
            torch.save({'unet': unet.state_dict(), 'ch': ch, 'res': a.res, 'alpha': float(alpha),
                        'steps': step, 'params_M': nparam}, ck)
            log(f'[save] {ck}')
    log('TRAIN_TINY_DONE')


if __name__ == '__main__':
    main()
