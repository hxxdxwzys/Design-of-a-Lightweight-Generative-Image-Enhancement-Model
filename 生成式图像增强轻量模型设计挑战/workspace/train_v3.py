"""v3 训练: 轻量一步潜扩散 + 对抗损失(PatchGAN) + 特征匹配 + 调参
GAN warmup (gan_start 步前仅 L1/LPIPS/L2)，之后加 hinge 对抗 + 特征匹配。
"""
import os
import sys
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
    ap.add_argument('--sd_path', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--unet_ch', type=str, default='64,128,256,512')
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--g_lr', type=float, default=1e-4)
    ap.add_argument('--d_lr', type=float, default=1e-4)
    ap.add_argument('--iters', type=int, default=15000)
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--w_pix', type=float, default=1.0)
    ap.add_argument('--w_lpips', type=float, default=1.0)
    ap.add_argument('--w_lat', type=float, default=0.5)
    ap.add_argument('--w_adv', type=float, default=0.05)
    ap.add_argument('--w_fm', type=float, default=1.0)
    ap.add_argument('--gan_start', type=int, default=2000)
    ap.add_argument('--resume', type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')
    unet_ch = tuple(int(x) for x in args.unet_ch.split(','))

    model = LightLatentDiff(args.sd_path, unet_ch=unet_ch).to(device)
    print(f'UNet params: {model.param_count()/1e6:.2f}M', flush=True)
    disc = PatchDiscriminator(in_ch=3, base=64).to(device)
    print(f'D params: {sum(p.numel() for p in disc.parameters())/1e6:.2f}M', flush=True)

    ds = HQDegradationDataset(args.data_dir, patch_size=args.patch_size)
    print(f'training images: {len(ds)}', flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    drop_last=True, pin_memory=True, persistent_workers=args.num_workers > 0)

    g_opt = torch.optim.AdamW(model.trainable_params(), lr=args.g_lr, betas=(0.9, 0.999), weight_decay=0)
    d_opt = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=args.iters)

    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    model.train()
    model.vae.eval()
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu')
        model.unet.load_state_dict(ck['unet'])
        start_step = ck.get('step', 0)
        for _ in range(start_step):
            g_sched.step()
        print(f'resumed from step {start_step}', flush=True)

    step = start_step
    t0 = time.time()
    zero = torch.tensor(0.0, device=device)
    while step < args.iters:
        for lq, hq in dl:
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)

            out, x0, lq_lat = model(lq, return_x0=True)
            with torch.no_grad():
                hq_lat = model.encode(hq)
            loss_pix = (out - hq).abs().mean()
            loss_lpips = lpips_fn(out.clamp(-1, 1), hq).mean()
            loss_lat = (x0 - hq_lat).pow(2).mean()
            g_loss = args.w_pix * loss_pix + args.w_lpips * loss_lpips + args.w_lat * loss_lat
            g_adv, fm, d_loss = zero, zero, zero

            if step >= args.gan_start:
                # ---- D step ----
                d_opt.zero_grad()
                real_logit, real_feats = disc(hq)
                fake_logit_det, _ = disc(out.detach())
                d_loss = F.relu(1 - real_logit).mean() + F.relu(1 + fake_logit_det).mean()
                d_loss.backward()
                d_opt.step()
                # ---- G adv + 特征匹配 ----
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

            if step % args.log_every == 0:
                el = time.time() - t0
                print(f'step {step}/{args.iters} g {g_loss.item():.4f} pix {loss_pix.item():.4f} '
                      f'lpips {loss_lpips.item():.4f} lat {loss_lat.item():.4f} '
                      f'adv {g_adv.item():.4f} fm {fm.item():.4f} d {d_loss.item():.4f} '
                      f'lr {g_sched.get_last_lr()[0]:.2e} {el/max(1,step-start_step):.2f}s/it', flush=True)
            if step % args.save_every == 0:
                torch.save({'unet': model.unet.state_dict(), 'disc': disc.state_dict(),
                            'step': step, 'args': vars(args), 'unet_ch': unet_ch},
                           os.path.join(args.output_dir, f'unet_step{step}.pth'))
                print(f'saved step {step}', flush=True)
            if step >= args.iters:
                break

    torch.save({'unet': model.unet.state_dict(), 'disc': disc.state_dict(),
                'step': step, 'args': vars(args), 'unet_ch': unet_ch},
               os.path.join(args.output_dir, 'unet_final.pth'))
    print('v3 training done', flush=True)


if __name__ == '__main__':
    main()
