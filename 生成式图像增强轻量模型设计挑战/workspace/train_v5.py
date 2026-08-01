"""v5 蒸馏训练: 学生(56M UNet, v3架构) 模仿 OSEDiff 教师输出。

损失: L1 + LPIPS(alex) + 潜空间L2, 全部对齐教师目标 (确定性目标, 无需 GAN)。
初始化: warm start 自 v3 unet ckpt (不丢已有能力, 收敛更快更稳)。
数据: gen_teacher_targets.py 离线生成的 (lq, hq_teacher) 对 (fp16 存盘)。
训练: fp32, AdamW lr 5e-5 cosine, patch512, 15k iters。
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
    """读 (xxx_lq.pt, xxx_hq.pt) 教师目标对, 返回 [-1,1] fp32。"""

    def __init__(self, target_dir):
        self.lq_paths = sorted(glob.glob(os.path.join(target_dir, '*_lq.pt')))
        print(f'teacher target pairs: {len(self.lq_paths)}', flush=True)
        assert len(self.lq_paths) > 0, f'no targets in {target_dir}'

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, idx):
        lq = torch.load(self.lq_paths[idx], map_location='cpu').float()
        hq = torch.load(self.lq_paths[idx].replace('_lq.pt', '_hq.pt'),
                        map_location='cpu').float()
        return lq, hq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sd_path', required=True)
    ap.add_argument('--target_dir', required=True, help='gen_teacher_targets 输出')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--init_ckpt', default=None, help='v3 unet ckpt (warm start)')
    ap.add_argument('--unet_ch', type=str, default='64,128,256,512')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--g_lr', type=float, default=5e-5)
    ap.add_argument('--iters', type=int, default=15000)
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--w_pix', type=float, default=1.0)
    ap.add_argument('--w_lpips', type=float, default=0.5)
    ap.add_argument('--w_lat', type=float, default=0.5)
    ap.add_argument('--resume', type=str, default=None)
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
        print(f'warm start from {args.init_ckpt} '
              f'(step={ck.get("step","?")})', flush=True)
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu')
        model.unet.load_state_dict(ck['unet'])
        start_step = ck.get('step', 0)
        print(f'resumed from step {start_step}', flush=True)

    ds = TeacherTargetDataset(args.target_dir)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True,
                                     pin_memory=True, persistent_workers=True)

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
        for lq, hq_t in dl:
            lq = lq.to(device, non_blocking=True)
            hq_t = hq_t.to(device, non_blocking=True)

            out, x0, _ = model(lq, return_x0=True)
            with torch.no_grad():
                hq_t_lat = model.encode(hq_t)

            loss_pix = (out - hq_t).abs().mean()
            loss_lpips = lpips_fn(out.clamp(-1, 1), hq_t).mean()
            loss_lat = (x0 - hq_t_lat).pow(2).mean()
            g_loss = (args.w_pix * loss_pix + args.w_lpips * loss_lpips
                      + args.w_lat * loss_lat)

            g_opt.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
            g_opt.step()
            g_sched.step()
            step += 1

            if step % args.log_every == 0:
                el = time.time() - t0
                print(f'step {step}/{args.iters} g {g_loss.item():.4f} '
                      f'pix {loss_pix.item():.4f} lpips {loss_lpips.item():.4f} '
                      f'lat {loss_lat.item():.4f} lr {g_sched.get_last_lr()[0]:.2e} '
                      f'{el/max(1, step-start_step):.2f}s/it', flush=True)
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
    print('v5 training done', flush=True)


if __name__ == '__main__':
    main()
