#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v5 蒸馏 (带 GT 锚定): 学生学 OSEDiff 教师目标 + GT 保真锚定。

损失 = w_t*(L1+LPIPS+L2 vs 教师目标) + w_gt*(L1+LPIPS vs GT锚定)
w_gt 拉住内容防漂移, w_t 提供感知方向。
warm start v3, LR 对齐 cosine (从 v3 训练终点继续)。
"""
import os
import sys
import glob
import argparse
import time
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.light_model import LightLatentDiff
import lpips


class TeacherTargetDataset(torch.utils.data.Dataset):
    """读 (lq, hq_t 教师目标, hq_gt GT锚定) 三件套。"""

    def __init__(self, target_dir):
        self.lq_paths = sorted(glob.glob(os.path.join(target_dir, '*_lq.pt')))
        print(f'teacher target pairs: {len(self.lq_paths)}', flush=True)
        assert len(self.lq_paths) > 0, f'no targets in {target_dir}'

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, idx):
        lq = torch.load(self.lq_paths[idx], map_location='cpu').float()
        base = self.lq_paths[idx].replace('_lq.pt', '')
        hq_t = torch.load(base + '_hq_t.pt', map_location='cpu').float()
        hq_gt = torch.load(base + '_hq_gt.pt', map_location='cpu').float()
        return lq, hq_t, hq_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sd_path', default='D:/models/sd21base')
    ap.add_argument('--target_dir', default='D:/models/teacher_targets_smoke')
    ap.add_argument('--output_dir', default='D:/models/runs/v5_gt_smoke')
    ap.add_argument('--init_ckpt', default='D:/models/unet_v3.pth', help='warm start v3')
    ap.add_argument('--unet_ch', type=str, default='64,128,256,512')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--g_lr', type=float, default=3e-5, help='低 LR 微调防破坏 v3')
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--w_t', type=float, default=0.5, help='教师目标权重')
    ap.add_argument('--w_gt', type=float, default=1.0, help='GT 锚定权重')
    ap.add_argument('--progress_file', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')
    unet_ch = tuple(int(x) for x in args.unet_ch.split(','))

    model = LightLatentDiff(args.sd_path, unet_ch=unet_ch).to(device)
    print(f'UNet params: {model.param_count()/1e6:.2f}M', flush=True)

    start_step = 0
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        ck = torch.load(args.init_ckpt, map_location='cpu')
        model.unet.load_state_dict(ck['unet'])
        print(f'warm start from {args.init_ckpt}', flush=True)
        # LR 对齐: v3 训完 iters, cosine 已到终点, 从这里继续 (低 LR 微调)
        # 用 T_max=args.iters 从头 cosine 但起始 lr 已很低 (args.g_lr 3e-5)

    ds = TeacherTargetDataset(args.target_dir)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True,
                                     pin_memory=True, persistent_workers=args.num_workers > 0)

    g_opt = torch.optim.AdamW(model.trainable_params(), lr=args.g_lr,
                              betas=(0.9, 0.999), weight_decay=0)
    g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=args.iters)
    for _ in range(start_step):
        g_sched.step()

    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    model.train()
    model.vae.eval()
    step = start_step
    t0 = time.time()
    while step < args.iters:
        for lq, hq_t, hq_gt in dl:
            lq = lq.to(device, non_blocking=True)
            hq_t = hq_t.to(device, non_blocking=True)
            hq_gt = hq_gt.to(device, non_blocking=True)

            out, x0, _ = model(lq, return_x0=True)
            with torch.no_grad():
                hq_t_lat = model.encode(hq_t)
                hq_gt_lat = model.encode(hq_gt)

            # 教师目标项
            loss_t_pix = (out - hq_t).abs().mean()
            loss_t_lpips = lpips_fn(out.clamp(-1, 1), hq_t).mean()
            loss_t_lat = (x0 - hq_t_lat).pow(2).mean()
            # GT 锚定项
            loss_g_pix = (out - hq_gt).abs().mean()
            loss_g_lpips = lpips_fn(out.clamp(-1, 1), hq_gt).mean()
            loss_g_lat = (x0 - hq_gt_lat).pow(2).mean()

            g_loss = (args.w_t * (loss_t_pix + loss_t_lpips + 0.5 * loss_t_lat)
                      + args.w_gt * (loss_g_pix + loss_g_lpips + 0.5 * loss_g_lat))

            g_opt.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
            g_opt.step()
            g_sched.step()
            step += 1

            if step % args.log_every == 0:
                el = time.time() - t0
                line = (f'step {step}/{args.iters} g {g_loss.item():.4f} '
                        f't_pix {loss_t_pix.item():.4f} t_lpips {loss_t_lpips.item():.4f} '
                        f'g_pix {loss_g_pix.item():.4f} g_lpips {loss_g_lpips.item():.4f} '
                        f'lr {g_sched.get_last_lr()[0]:.2e} {el/max(1, step-start_step):.2f}s/it')
                print(line, flush=True)
                if args.progress_file:
                    with open(os.path.join(args.output_dir, 'train_progress.txt'), 'a',
                              encoding='utf-8') as pf:
                        pf.write(line + '\n')
            if step % args.save_every == 0:
                torch.save({'unet': model.unet.state_dict(), 'step': step,
                            'args': vars(args), 'unet_ch': unet_ch},
                           os.path.join(args.output_dir, f'unet_step{step}.pth'))
                print(f'saved step {step}', flush=True)
            if step >= args.iters:
                break

    torch.save({'unet': model.unet.state_dict(), 'step': step,
                'args': vars(args), 'unet_ch': unet_ch},
               os.path.join(args.output_dir, 'unet_final.pth'))
    print('v5_gt training done', flush=True)


if __name__ == '__main__':
    main()
