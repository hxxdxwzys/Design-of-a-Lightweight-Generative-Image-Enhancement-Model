"""训练轻量一步潜扩散 (fp32 稳定训练)
损失 = 像素L1 + LPIPS + 潜空间L2
"""
import os
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.light_model import LightLatentDiff
from data.dataset import HQDegradationDataset
import lpips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sd_path', required=True, help='SD2.1-base 路径(含 vae/ 子目录)')
    ap.add_argument('--data_dir', required=True, help='HQ 训练图目录')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--unet_ch', type=str, default='64,128,256,512')
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--iters', type=int, default=50000)
    ap.add_argument('--save_every', type=int, default=2000)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--w_pix', type=float, default=1.0)
    ap.add_argument('--w_lpips', type=float, default=1.0)
    ap.add_argument('--w_lat', type=float, default=0.5)
    ap.add_argument('--resume', type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')
    unet_ch = tuple(int(x) for x in args.unet_ch.split(','))

    model = LightLatentDiff(args.sd_path, unet_ch=unet_ch).to(device)
    print(f'UNet params: {model.param_count()/1e6:.2f}M', flush=True)

    ds = HQDegradationDataset(args.data_dir, patch_size=args.patch_size)
    print(f'training images: {len(ds)}', flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True, pin_memory=True,
                    persistent_workers=args.num_workers > 0)

    opt = torch.optim.AdamW(model.trainable_params(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu')
        model.unet.load_state_dict(ck['unet'])
        start_step = ck.get('step', 0)
        for _ in range(start_step):
            sched.step()
        print(f'resumed from step {start_step}', flush=True)

    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    model.train()
    model.vae.eval()  # 冻结 VAE 保持 eval

    step = start_step
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
            loss = args.w_pix * loss_pix + args.w_lpips * loss_lpips + args.w_lat * loss_lat

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % args.log_every == 0:
                el = time.time() - t0
                print(f'step {step}/{args.iters} loss {loss.item():.4f} '
                      f'pix {loss_pix.item():.4f} lpips {loss_lpips.item():.4f} '
                      f'lat {loss_lat.item():.4f} lr {sched.get_last_lr()[0]:.2e} '
                      f'{el/max(1,step-start_step):.2f}s/it', flush=True)
            if step % args.save_every == 0:
                ckpt = {'unet': model.unet.state_dict(), 'step': step,
                        'args': vars(args), 'unet_ch': unet_ch}
                torch.save(ckpt, os.path.join(args.output_dir, f'unet_step{step}.pth'))
                print(f'saved checkpoint step {step}', flush=True)
            if step >= args.iters:
                break

    torch.save({'unet': model.unet.state_dict(), 'step': step,
                'args': vars(args), 'unet_ch': unet_ch},
               os.path.join(args.output_dir, 'unet_final.pth'))
    print('training done', flush=True)


if __name__ == '__main__':
    main()
