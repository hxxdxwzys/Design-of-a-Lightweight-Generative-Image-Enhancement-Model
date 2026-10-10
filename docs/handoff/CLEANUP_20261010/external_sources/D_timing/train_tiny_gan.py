# -*- coding: utf-8 -*-
"""微模型 v7: 蒸馏 M24 观感 + 对抗损失 (2026-09-14)
诊断依据: L1+LPIPS 蒸馏只能学到"幻觉目标的均值"= 更模糊 (v5 lapvar 127 vs 目标 602, clipiqa 0.39 vs 0.90)
         且单纯锐化抬不动 clipiqa => 必须让模型自己【生成】纹理分布 => 加 PatchGAN
配置: 256 域 (graph 后 ~50x), 0.87M G + ~0.5M D, L1 + 0.2*LPIPS + 0.1*LSGAN
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault('HF_HUB_OFFLINE', '1')
OUT = Path(r'D:/models/tiny_v1'); OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / 'train_gan.log'


def log(m):
    s = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


def band_limit(x):
    """去掉高频(清晰度)线索: 2x 平均池化再最近邻放大回原尺寸"""
    return F.interpolate(F.avg_pool2d(x, 2), size=x.shape[-2:], mode='nearest')


class PatchD(nn.Module):
    """小 PatchGAN 判别器 (256 域)"""

    def __init__(self, ch=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for i in range(1, n_layers):
            prev = mult
            mult = min(2 ** i, 8)
            layers += [nn.Conv2d(ch * prev, ch * mult, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(ch * mult), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ch * mult, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


LQ_F = []; TGT_F = []; POOL = ThreadPoolExecutor(max_workers=10)


def load_pair(_i):
    k = np.random.randint(len(LQ_F))
    for _ in range(6):
        lq = np.asarray(Image.open(LQ_F[k]).convert('RGB'), dtype=np.uint8)
        tgt = np.asarray(Image.open(TGT_F[k]).convert('RGB'), dtype=np.uint8)
        if lq.shape == tgt.shape:
            return tgt, lq
        k = np.random.randint(len(LQ_F))
    raise RuntimeError('尺寸不一致')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distill_dir', default=r'D:/datasets/distill256')
    ap.add_argument('--res', type=int, default=256)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lr_d', type=float, default=2e-4)
    ap.add_argument('--ch', default='32,32,32,64')
    ap.add_argument('--w_lpips', type=float, default=0.2)
    ap.add_argument('--w_gan', type=float, default=0.1)
    ap.add_argument('--tag', default='v7gan')
    ap.add_argument('--loss_res', type=int, default=0)   # 0=同 res; 512 => 学生在 res 跑, 但损失/判别器在 512 域 (学会补偿上采样模糊)
    ap.add_argument('--init_from', default='')          # 从 ckpt 初始化 G (断点恢复; 优化器/调度从头)
    ap.add_argument('--d_band', type=int, default=0)     # 1 => 判别器输入先做 band-limit(avgpool2+nearest 上采样),
                                                         #      去掉"清晰度"这个平凡线索, 让 D 只判结构/伪影 (修 D 秒杀退化)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    globals()['LQ_F'] = sorted((Path(a.distill_dir) / 'lq').glob('*.jpg'))
    globals()['TGT_F'] = sorted((Path(a.distill_dir) / 'tgt').glob('*.jpg'))
    assert len(LQ_F) == len(TGT_F) > 0
    log(f'[data] {len(LQ_F)} 对 (256 域蒸馏对)')
    dev = 'cuda'
    torch.manual_seed(0); np.random.seed(0)
    ch = tuple(int(x) for x in a.ch.split(','))

    from diffusers import UNet2DModel, DDPMScheduler
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear',
                          beta_start=0.00085, beta_end=0.012)
    alpha = sched.alphas_cumprod[500].to(dev).float()
    sq_a, sq_1a = alpha.sqrt(), (1 - alpha).sqrt()
    G = UNet2DModel(sample_size=a.res, in_channels=3, out_channels=3, layers_per_block=1,
                    block_out_channels=ch, down_block_types=('DownBlock2D',) * len(ch),
                    up_block_types=('UpBlock2D',) * len(ch)).to(dev)
    if a.init_from and Path(a.init_from).exists():
        d0 = torch.load(a.init_from, map_location='cpu', weights_only=False)
        miss = G.load_state_dict(d0['unet'], strict=False)
        log('[init] G 从 %s 初始化 (该 ckpt step=%s, missing=%d)' % (Path(a.init_from).name, d0.get('steps'), len(miss.missing_keys)))
    D = PatchD().to(dev)
    log(f'[model] G={sum(p.numel() for p in G.parameters())/1e6:.2f}M '
        f'D={sum(p.numel() for p in D.parameters())/1e6:.2f}M res={a.res} bs={a.bs} w_gan={a.w_gan}')
    import lpips as lpips_mod
    lp = lpips_mod.LPIPS(net='alex', verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    optG = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.9, 0.99))
    optD = torch.optim.AdamW(D.parameters(), lr=a.lr_d, betas=(0.9, 0.99))
    schG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=a.steps, eta_min=a.lr * 0.05)
    schD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=a.steps, eta_min=a.lr_d * 0.05)
    scaler = torch.amp.GradScaler('cuda')

    t0 = time.time(); runG = runD = 0.0
    for step in range(1, a.steps + 1):
        pairs = list(POOL.map(load_pair, range(a.bs)))
        hq = torch.from_numpy(np.stack([p[0] for p in pairs])).permute(0, 3, 1, 2).to(dev).float().div_(127.5).sub_(1)
        lq = torch.from_numpy(np.stack([p[1] for p in pairs])).permute(0, 3, 1, 2).to(dev).float().div_(127.5).sub_(1)
        hq_full = hq                      # 原始域 (蒸馏对存的是 512) -> 损失可用
        if hq.shape[-1] != a.res:   # 降到学生实际运行的域
            hq = F.interpolate(hq, size=(a.res, a.res), mode='bicubic', align_corners=False)
            lq = F.interpolate(lq, size=(a.res, a.res), mode='bicubic', align_corners=False)
        LRES = a.loss_res or a.res

        # ---- 单次 G 前向, D 用 detach 复用 (省一次前向 + 显存) ----
        with torch.autocast('cuda', dtype=torch.float16):
            pred = G(lq, torch.full((a.bs,), 500, dtype=torch.long, device=dev)).sample
            x0 = (lq - sq_1a * pred.float()) / sq_a
            x0c = x0.clamp(-1, 1)
            if LRES != a.res:   # 把学生输出与教师目标都放到 LRES 域再算损失 => 学生学会抵消上采样模糊
                x0l = F.interpolate(x0c, size=(LRES, LRES), mode='bicubic', align_corners=False)
                hql = F.interpolate(hq_full, size=(LRES, LRES), mode='bicubic', align_corners=False)
            else:
                x0l, hql = x0c, hq
            loss_l1 = F.l1_loss(x0l, hql)
            loss_lp = lp(x0l, hql).mean()
            _dg = band_limit(x0l) if a.d_band else x0l
            loss_gan = ((D(_dg) - 1) ** 2).mean()
            lossG = loss_l1 + a.w_lpips * loss_lp + a.w_gan * loss_gan
        optG.zero_grad(set_to_none=True)
        scaler.scale(lossG).backward()
        scaler.unscale_(optG)
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        scaler.step(optG); scaler.update(); schG.step()

        # ---- D (用 detach 的假图) ----
        with torch.autocast('cuda', dtype=torch.float16):
            _dr = band_limit(hql) if a.d_band else hql
            _df = band_limit(x0l.detach()) if a.d_band else x0l.detach()
            d_real = D(_dr); d_fake = D(_df)
            lossD = 0.5 * ((d_real - 1) ** 2).mean() + 0.5 * (d_fake ** 2).mean()
        optD.zero_grad(set_to_none=True)
        scaler.scale(lossD).backward(); scaler.step(optD); scaler.update(); schD.step()

        runG = 0.9 * runG + 0.1 * float(lossG) if step > 1 else float(lossG)
        runD = 0.9 * runD + 0.1 * float(lossD) if step > 1 else float(lossD)
        if step % 100 == 0:
            log(f'step {step}/{a.steps} G={runG:.3f} (l1={float(loss_l1):.3f} lp={float(loss_lp):.3f} gan={float(loss_gan):.3f}) '
                f'D={runD:.3f} {(time.time()-t0)/step:.2f}s/step')
        if step % 2000 == 0 or step == a.steps:
            torch.save({'unet': G.state_dict(), 'ch': ch, 'res': a.res, 'alpha': float(alpha),
                        'steps': step, 'gan': True}, OUT / f'{a.tag}_s{step}.pt')
            log(f'[save] {a.tag}_s{step}.pt')
    log('TRAIN_GAN_DONE')


if __name__ == '__main__':
    main()
