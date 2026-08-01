"""本地 4060 (8GB) 训练: v3+ 逻辑 (GAN强化 + R1 + EMA) + fp32 + 小patch适配。

本地 8GB 显存方案 (实测):
  - fp32 全精度 (与云端 v3 一致, 稳定无 NaN; fp16 在连续训练时权重溢出)
  - patch192 batch4: 峰值 4.3G, 余量给 GAN+LPIPS (~6G)
  - warm start 自 v3 权重 (D:/models/unet_v3.pth)
速度: fp32 patch192 batch4 约 1.2s/it -> 5000 iters ~1.7h
"""
import os
import sys
import copy
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.light_model import LightLatentDiff
from models.discriminator import PatchDiscriminator
from data.dataset import HQDegradationDataset
import lpips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sd_path', default='D:/models/sd21base')
    ap.add_argument('--data_dir', default='D:/datasets/div2k')
    ap.add_argument('--output_dir', default='D:/models/runs/v3plus_local')
    ap.add_argument('--resume', default='D:/models/unet_v3.pth', help='warm start v3')
    ap.add_argument('--unet_ch', type=str, default='64,128,256,512')
    ap.add_argument('--patch_size', type=int, default=192)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--g_lr', type=float, default=1e-4)
    ap.add_argument('--d_lr', type=float, default=1e-4)
    ap.add_argument('--iters', type=int, default=5000)
    ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--w_pix', type=float, default=1.0)
    ap.add_argument('--w_lpips', type=float, default=1.0)
    ap.add_argument('--w_lat', type=float, default=0.5)
    ap.add_argument('--w_adv', type=float, default=0.15)
    ap.add_argument('--w_fm', type=float, default=1.0)
    ap.add_argument('--r1_lambda', type=float, default=0.5)
    ap.add_argument('--gan_start', type=int, default=1500)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')
    unet_ch = tuple(int(x) for x in args.unet_ch.split(','))

    model = LightLatentDiff(args.sd_path, unet_ch=unet_ch).to(device)
    disc = PatchDiscriminator(in_ch=3, base=64).to(device)
    print(f'UNet params: {model.param_count()/1e6:.2f}M | D: '
          f'{sum(p.numel() for p in disc.parameters())/1e6:.2f}M', flush=True)

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu')
        if 'unet' in ck:
            model.unet.load_state_dict(ck['unet'])
        print(f'warm start from {args.resume}', flush=True)

    ema_unet = copy.deepcopy(model.unet).to(device)
    ema_unet.requires_grad_(False)

    ds = HQDegradationDataset(args.data_dir, patch_size=args.patch_size)
    print(f'training images: {len(ds)}', flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.num_workers > 0)

    g_opt = torch.optim.AdamW(model.trainable_params(), lr=args.g_lr, betas=(0.9, 0.999), weight_decay=0)
    d_opt = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=args.iters)

    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    model.train()
    model.vae.eval()
    step = 0
    t0 = time.time()
    zero = torch.tensor(0.0, device=device)
    while step < args.iters:
        for lq, hq in dl:
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)

            out, x0, _ = model(lq, return_x0=True)
            with torch.no_grad():
                hq_lat = model.encode(hq)
            loss_pix = (out - hq).abs().mean()
            loss_lpips = lpips_fn(out.clamp(-1, 1), hq).mean()
            loss_lat = (x0 - hq_lat).pow(2).mean()
            g_loss = args.w_pix * loss_pix + args.w_lpips * loss_lpips + args.w_lat * loss_lat
            g_adv, fm, d_loss = zero, zero, zero

            if step >= args.gan_start:
                # D step (hinge + R1)
                d_opt.zero_grad()
                hq.requires_grad_(True)
                real_logit, real_feats = disc(hq)
                fake_logit_det, _ = disc(out.detach())
                d_loss = F.relu(1 - real_logit).mean() + F.relu(1 + fake_logit_det).mean()
                if args.r1_lambda > 0:
                    grad_real = torch.autograd.grad(
                        outputs=real_logit.sum(), inputs=hq,
                        create_graph=True, retain_graph=True)[0]
                    r1 = grad_real.pow(2).view(hq.size(0), -1).sum(1).mean()
                    d_loss = d_loss + args.r1_lambda * r1
                d_loss.backward()
                d_opt.step()
                hq.requires_grad_(False)
                # G adv + 特征匹配
                fake_logit, fake_feats = disc(out)
                g_adv = -fake_logit.mean()
                fm = sum(F.l1_loss(f, r.detach()) for f, r in zip(fake_feats, real_feats)) / len(fake_feats)
                g_loss = g_loss + args.w_adv * g_adv + args.w_fm * fm

            g_opt.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
            g_opt.step()
            g_sched.step()
            step += 1

            with torch.no_grad():
                for ep, p in zip(ema_unet.parameters(), model.unet.parameters()):
                    ep.data.mul_(args.ema_decay).add_(p.data, alpha=1 - args.ema_decay)

            if step % args.log_every == 0:
                el = time.time() - t0
                print(f'step {step}/{args.iters} g {g_loss.item():.4f} '
                      f'pix {loss_pix.item():.4f} lpips {loss_lpips.item():.4f} '
                      f'lat {loss_lat.item():.4f} adv {g_adv.item():.4f} '
                      f'fm {fm.item():.4f} d {d_loss.item():.4f} '
                      f'{el/max(1, step):.2f}s/it', flush=True)
            if step % args.save_every == 0:
                torch.save({'unet': model.unet.state_dict(), 'ema_unet': ema_unet.state_dict(),
                            'disc': disc.state_dict(), 'step': step, 'args': vars(args),
                            'unet_ch': unet_ch},
                           os.path.join(args.output_dir, f'unet_step{step}.pth'))
                print(f'saved step {step}', flush=True)
            if step >= args.iters:
                break

    torch.save({'unet': model.unet.state_dict(), 'ema_unet': ema_unet.state_dict(),
                'disc': disc.state_dict(), 'step': step, 'args': vars(args), 'unet_ch': unet_ch},
               os.path.join(args.output_dir, 'unet_final.pth'))
    print('train_local done', flush=True)


if __name__ == '__main__':
    main()
