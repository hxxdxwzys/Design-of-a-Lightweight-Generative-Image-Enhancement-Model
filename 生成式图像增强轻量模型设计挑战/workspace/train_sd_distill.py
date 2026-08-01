"""SD 教师感知蒸馏训练 (v5 候选): 学生(56M, v3 warm start) 向 sd21base UNet 的感知先验蒸馏。

核心思想 (SDS-lite / 单步教师重建):
  学生输出 latent z0 (LQ->HQ 的一步映射)
  -> 加噪 z_t = sqrt(a)*z0 + sqrt(1-a)*eps  (t ~ Uniform(200,900))
  -> 教师 SD UNet 预测 eps -> 一步重建 x0_t (教师认为"真实"的方向)
  -> 损失 ||z0 - x0_t.detach()||^2 * w(t)   (学生被拉向 SD 先验)
  并行保真损失 (L1 + LPIPS + 潜L2 vs GT) 防止 v4 教训(内容丢失)

教师侧: sd21base UNet(frozen, fp16) + CLIP text encoder (prompt 条件)
        + RAM tags (可选, 默认固定 prompt 先跑通)
学生侧: LightLatentDiff (v3 架构), warm start 自 v3 权重, EMA 推理
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
from data.dataset import HQDegradationDataset
import lpips

PROMPT = 'a high-quality, detailed, sharp, clean photograph'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sd_path', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--resume', default=None, help='warm start unet (v3 unet_v3.pth)')
    ap.add_argument('--unet_ch', type=str, default='64,128,256,512')
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--g_lr', type=float, default=1e-4)
    ap.add_argument('--iters', type=int, default=15000)
    ap.add_argument('--save_every', type=int, default=1000)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--w_distill', type=float, default=1.0)
    ap.add_argument('--w_pix', type=float, default=0.5)
    ap.add_argument('--w_lpips', type=float, default=0.5)
    ap.add_argument('--w_lat', type=float, default=0.5)
    ap.add_argument('--t_min', type=int, default=200)
    ap.add_argument('--t_max', type=int, default=900)
    ap.add_argument('--cfg_scale', type=float, default=1.0,
                    help='teacher 分类器自由引导; 1.0=纯cond, >1 加 uncond 反相')
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--prompt', type=str, default=PROMPT)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')
    unet_ch = tuple(int(x) for x in args.unet_ch.split(','))
    assert args.t_min < args.t_max <= 1000

    # ---- 学生 (v3 warm start) ----
    model = LightLatentDiff(args.sd_path, unet_ch=unet_ch).to(device)
    print(f'student UNet params: {model.param_count()/1e6:.2f}M', flush=True)
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu')
        if 'unet' in ck:
            model.unet.load_state_dict(ck['unet'])
        print(f'warm start from {args.resume}', flush=True)
    ema_unet = copy.deepcopy(model.unet).to(device)
    ema_unet.requires_grad_(False)

    # ---- 教师 (SD2.1 UNet + CLIP, frozen) ----
    from diffusers import UNet2DConditionModel, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    teacher = UNet2DConditionModel.from_pretrained(
        args.sd_path, subfolder='unet').to(device).eval()
    teacher.requires_grad_(False)
    teacher = teacher.half() if args.cfg_scale <= 1.0 else teacher
    scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder='scheduler')
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd_path, subfolder='text_encoder').to(device).eval()
    text_encoder.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_path, subfolder='tokenizer')
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    print(f'teacher UNet params: {sum(p.numel() for p in teacher.parameters())/1e6:.0f}M '
          f'fp16={teacher.dtype}', flush=True)

    def encode_prompt(text, neg=False):
        tok = tokenizer(text, max_length=77, padding='max_length',
                        truncation=True, return_tensors='pt')
        emb = text_encoder(tok.input_ids.to(device))[0]
        return emb.half() if teacher.dtype == torch.float16 else emb

    prompt_embeds = encode_prompt(args.prompt)  # [1,77,1024]
    uncond_embeds = encode_prompt('') if args.cfg_scale > 1.0 else None

    # ---- 数据 ----
    ds = HQDegradationDataset(args.data_dir, patch_size=args.patch_size)
    print(f'training images: {len(ds)}', flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True,
                    pin_memory=True, persistent_workers=True)

    g_opt = torch.optim.AdamW(model.trainable_params(), lr=args.g_lr,
                              betas=(0.9, 0.999), weight_decay=0)
    g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=args.iters)

    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    model.train()
    model.vae.eval()
    step = 0
    t0 = time.time()
    while step < args.iters:
        for lq, hq in dl:
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)
            B = lq.size(0)

            # 学生前向 (返回 latent x0 与像素 out)
            out, z0, _ = model(lq, return_x0=True)
            with torch.no_grad():
                hq_lat = model.encode(hq)

            # ---- 蒸馏损失 (SDS-lite) ----
            t = torch.randint(args.t_min, args.t_max, (B,), device=device)
            noise = torch.randn_like(z0)
            a = alphas_cumprod[t]  # [B]
            sigma = (1 - a).sqrt()
            zt = a.sqrt().view(B, 1, 1, 1) * z0 + sigma.view(B, 1, 1, 1) * noise
            cond = prompt_embeds.expand(B, -1, -1)
            if teacher.dtype == torch.float16:
                zt_h = zt.half()
            else:
                zt_h = zt
            with torch.no_grad():
                eps_pred = teacher(zt_h, t, encoder_hidden_states=cond).sample
                if uncond_embeds is not None:
                    eps_un = teacher(zt_h, t,
                                     encoder_hidden_states=uncond_embeds.expand(B, -1, -1)).sample
                    eps_pred = eps_pred + args.cfg_scale * (eps_pred - eps_un)
                eps_pred = eps_pred.float()
                x0_t = (zt - sigma.view(B, 1, 1, 1) * eps_pred) / a.sqrt().view(B, 1, 1, 1)
                w = (1 - a)  # 感知权重: 大 t 高权重
            loss_distill = ((z0 - x0_t) ** 2).mul(w.view(B, 1, 1, 1)).mean()

            # ---- 保真损失 ----
            loss_pix = (out - hq).abs().mean()
            loss_lpips = lpips_fn(out.clamp(-1, 1), hq).mean()
            loss_lat = (z0 - hq_lat).pow(2).mean()
            g_loss = (args.w_distill * loss_distill + args.w_pix * loss_pix
                      + args.w_lpips * loss_lpips + args.w_lat * loss_lat)

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
                      f'distill {loss_distill.item():.4f} pix {loss_pix.item():.4f} '
                      f'lpips {loss_lpips.item():.4f} lat {loss_lat.item():.4f} '
                      f't={t.mean().item():.0f} lr {g_sched.get_last_lr()[0]:.2e} '
                      f'{el/max(1, step):.2f}s/it', flush=True)
            if step % args.save_every == 0:
                torch.save({'unet': model.unet.state_dict(),
                            'ema_unet': ema_unet.state_dict(),
                            'step': step, 'args': vars(args), 'unet_ch': unet_ch},
                           os.path.join(args.output_dir, f'unet_step{step}.pth'))
                print(f'saved step {step}', flush=True)
            if step >= args.iters:
                break

    torch.save({'unet': model.unet.state_dict(), 'ema_unet': ema_unet.state_dict(),
                'step': step, 'args': vars(args), 'unet_ch': unet_ch},
               os.path.join(args.output_dir, 'unet_final.pth'))
    print('sd_distill training done', flush=True)


if __name__ == '__main__':
    main()
