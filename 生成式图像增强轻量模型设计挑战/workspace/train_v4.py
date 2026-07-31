"""v4 训练: LoRA 微调 SD2.1 UNet (预训练先验) + PatchGAN 对抗 + 特征匹配。
VAE/CLIP/UNet基座冻结, 只训 LoRA+conv_in。fp32 + 梯度检查点(V100 32GB)。
"""
import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.light_model_v4 import LightLatentDiffV4
from models.discriminator import PatchDiscriminator
from data.dataset import HQDegradationDataset
import lpips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sd_path', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--lora_rank', type=int, default=16)
    ap.add_argument('--patch_size', type=int, default=512)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--g_lr', type=float, default=1e-4)
    ap.add_argument('--d_lr', type=float, default=1e-4)
    ap.add_argument('--iters', type=int, default=10000)
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--w_pix', type=float, default=1.0)
    ap.add_argument('--w_lpips', type=float, default=1.0)
    ap.add_argument('--w_lat', type=float, default=0.5)
    ap.add_argument('--w_adv', type=float, default=0.05)
    ap.add_argument('--w_fm', type=float, default=1.0)
    ap.add_argument('--gan_start', type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')

    model = LightLatentDiffV4(args.sd_path, lora_rank=args.lora_rank).to(device)
    print(f'V4 trainable params: {model.param_count()/1e6:.2f}M (LoRA rank {args.lora_rank} + conv_in)', flush=True)
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
    model.vae.eval(); model.text_encoder.eval()
    zero = torch.tensor(0.0, device=device)
    step = 0
    t0 = time.time()
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
                d_opt.zero_grad()
                real_logit, real_feats = disc(hq)
                fake_logit_det, _ = disc(out.detach())
                d_loss = F.relu(1 - real_logit).mean() + F.relu(1 + fake_logit_det).mean()
                d_loss.backward()
                d_opt.step()
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
                      f'lr {g_sched.get_last_lr()[0]:.2e} {el/step:.2f}s/it', flush=True)
            if step % args.save_every == 0:
                sd = {n: p for n, p in model.unet.named_parameters() if p.requires_grad}
                torch.save({'lora_state': sd, 'step': step, 'args': vars(args),
                            'lora_rank': args.lora_rank},
                           os.path.join(args.output_dir, f'v4_step{step}.pth'))
                print(f'saved step {step}', flush=True)
            if step >= args.iters:
                break

    sd = {n: p for n, p in model.unet.named_parameters() if p.requires_grad}
    torch.save({'lora_state': sd, 'step': step, 'args': vars(args), 'lora_rank': args.lora_rank},
               os.path.join(args.output_dir, 'v4_final.pth'))
    print('v4 training done', flush=True)


if __name__ == '__main__':
    main()
