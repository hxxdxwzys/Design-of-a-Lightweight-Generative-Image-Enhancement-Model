# -*- coding: utf-8 -*-
"""B: VOSR 0.5B os LoRA 感知微调 (2026-08-18 初版 / 2026-09-02 P100 服务器版)

目标: 在 VOSR 一步生成流形内, 把输出推向更高 4K clipiqa (感知), LPIPS 锚定保内容。
数据: DF2K + EmpProfile (测试集实测 sigma/noise/jpeg 三元组 bootstrap, 画像匹配)
损失: LPIPS(as_loss=True) + (1-clipiqa)(as_loss=True) + L1, 权重按变体
变体: --w-clip 控制感知强度 (0.3/0.8/1.5), 测试集直选定胜负

P100 服务器版新增 (2026-09-02):
  - 路径全部环境变量覆盖 (VOSR_DIR/VOSR_CKPT/VOSR_AE/VOSR_LWDEC/VOSR_CACHE/VOSR_DATA/VOSR_OUT/EMP_DEG/EMP_JPEG/PAUSE_FLAG)
  - --half: 基座 fp16 (P100 16GB 跑 patch1024 需要; peft LoRA 保持 fp32)
  - 全状态续训 (opt/sched/rng 落盘 + --resume-from 无缝恢复, cosine 轨迹连续)
  - 暂停协议: PAUSE_FLAG 文件存在 → 每 25 步存 lora_pause.pt 干净退出

用法 (本地): D:/Python/python.exe train_vosr_lora.py --variant v1 --w-clip 0.8 [--steps 3000]
用法 (P100): CUDA_VISIBLE_DEVICES=0 python train_vosr_lora.py --variant p1024_w20 --w-clip 2.0 \
      --steps 2500 --save-every 500 --patch 1024 --half
输出: {VOSR_OUT}/{variant}/lora_s{step}.pt
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'  # Windows 无 triton: torch.compile 静默回退 eager

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

VOSR_DIR = os.environ.get('VOSR_DIR', r'D:/models/finals/VOSR')
sys.path.insert(0, VOSR_DIR)
sys.path.insert(0, r'D:/tmp_timing')

import pyiqa  # noqa: E402
from diffusers import AutoencoderKL  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from models.lightningdit import LightningDiT  # noqa: E402
from models.light_decoder import LightDecoder  # noqa: E402
from train_v2 import EmpProfile  # noqa: E402 (__main__ 保护已确认)

CKPT = os.environ.get('VOSR_CKPT', r'C:/models/finals/preset/ckpts/VOSR_0.5B_os')
AE = os.environ.get('VOSR_AE', r'C:/models/finals/preset/ckpts/stable-diffusion-2-1-base')
LWDEC = os.environ.get('VOSR_LWDEC', r'C:/models/finals/preset/ckpts/sd21_lwdecoder.pth')
TORCH_CACHE = os.environ.get('VOSR_CACHE', r'C:/models/finals/preset/ckpts/torch_cache')
DATA_DIRS = os.environ.get('VOSR_DATA', ';'.join([r'D:/datasets/div2k/pngtxt_ori', r'D:/datasets/flickr2k'])).split(';')
OUT = os.environ.get('VOSR_OUT', r'D:/models/vosr_lora')
EMP_DEG = os.environ.get('EMP_DEG', r'D:/tmp_timing/degradation_analysis.json')
EMP_JPEG = os.environ.get('EMP_JPEG', r'D:/tmp_timing/jpeg_blockiness.json')
PAUSE_FLAG = Path(os.environ.get('PAUSE_FLAG', r'D:/tmp_timing/v14_pause.flag'))

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_base_model(args):
    torch.hub.set_dir(TORCH_CACHE)
    model = LightningDiT(
        input_size=args.resolution // 8,
        patch_size=args.patch_size,
        in_channels=2 * (4 if args.ae_type == 'sd2' else 16),
        out_channels=4 if args.ae_type == 'sd2' else 16,
        hidden_size=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        z_dims=args.enc_dim,
        encdim_ratio=args.encdim_ratio,
        auxiliary_time_cond=args.auxiliary_time_cond,
        use_qknorm=args.use_qknorm,
        use_swiglu=args.use_swiglu,
        use_rope=args.use_rope,
        use_rmsnorm=args.use_rmsnorm,
        wo_shift=args.wo_shift,
        num_fused_layers=len(args.layer_dinov2b_list),
    )
    wpath = Path(CKPT) / 'checkpoints' / 'ema_model.safetensors'
    sd = load_file(str(wpath))
    model.load_state_dict(sd, strict=False)
    model.forward = model.forward_flexible
    model.eval()
    return model


def load_venc(args, device):
    torch.hub.set_dir(TORCH_CACHE)
    import types
    # 国内服务器 GitHub 被墙: 优先本地 hub 缓存仓库 (source='local', 完全离线)
    local_repo = os.path.join(TORCH_CACHE, 'facebookresearch_dinov2_main')
    if os.path.isdir(local_repo):
        try:
            encoder = torch.hub.load(local_repo, 'dinov2_vitb14', source='local')
            print('[venc] dinov2_vitb14 loaded from local hub cache (offline)', flush=True)
        except Exception as e:
            print(f'[warn] local dinov2 load failed ({e}); fallback remote torch.hub', flush=True)
            encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    else:
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    del encoder.head
    encoder.head = torch.nn.Identity()

    def forward_with_features(self, x, masks=None):
        features = {}
        layer_indices = list(range(len(self.blocks)))
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in layer_indices:
                features[f'layer_{i}'] = x[:, 1:]
        x_norm = self.norm(x)
        return features, x_norm[:, 1:]

    encoder.forward_with_features = types.MethodType(forward_with_features, encoder)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


@torch.no_grad()
def venc_features(venc, lq_tensor, args):
    """lq_tensor [-1,1] -> z 列表 (detached)"""
    raw_image = (0.5 * lq_tensor + 0.5) * 255
    x = F.interpolate(raw_image, args.dinov2_size, mode='bicubic').clip(0., 255.) / 255.
    x = (x - torch.tensor(IMAGENET_DEFAULT_MEAN, device=x.device).view(1, 3, 1, 1)) / \
        torch.tensor(IMAGENET_DEFAULT_STD, device=x.device).view(1, 3, 1, 1)
    features, x_norm = venc.forward_with_features(x)
    z = [v for k, v in features.items() if k.startswith('layer_')]
    z[-1] = x_norm
    z = [z[i] for i in args.layer_dinov2b_list]
    return z


# peft 0.20: target_modules 构造可用但 to_dict 不序列化 (target_parameters 是参数级机制, 不适用)
LORA_TARGETS = ['qkv', 'proj', 'w12', 'w3', 'fc1', 'fc2', 'linear']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--w-clip', type=float, required=True)
    ap.add_argument('--w-lpips', type=float, default=1.0)
    ap.add_argument('--w-l1', type=float, default=0.05)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--alpha', type=int, default=64)
    ap.add_argument('--patch', type=int, default=512)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--half', action='store_true', help='基座 fp16 (P100 16GB 跑大 patch 用; LoRA 仍 fp32)')
    ap.add_argument('--resume-from', default=None, help='从已有 LoRA ckpt 继续训练 (全状态续训/精修)')
    ap.add_argument('--ckpt-prefix', default='lora', help='保存文件名前缀 (精修用 ref)')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = 'cuda'

    # 基础配置
    base = json.load(open(os.path.join(CKPT, 'args.json')))
    base['auxiliary_time_cond'] = True
    A = argparse.Namespace(**base)

    out_dir = Path(OUT) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = out_dir / 'train.log'
    def log(msg):
        line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        with logf.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    log(f'[boot] variant={args.variant} w_clip={args.w_clip} steps={args.steps} lr={args.lr} '
        f'rank={args.rank} patch={args.patch} half={args.half}')

    # 模型 + LoRA
    model = load_base_model(A).to(device)
    if args.half:
        model = model.half()
    if args.resume_from:
        prev = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        cfg = json.loads(prev['lora_config'])
        lora_cfg = LoraConfig(
            r=cfg['r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg.get('lora_dropout', 0.0),
            bias=cfg.get('bias', 'none'), target_modules=cfg['target_modules'],
        )
        log(f'[resume] {args.resume_from} (step={prev.get("step")})')
    else:
        lora_cfg = LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias='none',
            target_modules=LORA_TARGETS,
        )
    model = get_peft_model(model, lora_cfg)
    if args.resume_from:
        lora_sd = {k: v for k, v in prev['state_dict'].items() if 'lora' in k}
        missing, unexpected = model.load_state_dict(lora_sd, strict=False)
        log(f'[resume] loaded {len(lora_sd)} keys, missing={len(missing)} unexpected={len(unexpected)}')
    model.print_trainable_parameters()

    # VAE + LightDecoder + DINOv2 (冻结)
    vae = AutoencoderKL.from_pretrained(AE, subfolder='vae').to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    lw = torch.load(LWDEC, map_location='cpu', weights_only=False)
    dec = LightDecoder(**{k: lw['config'][k] for k in ('in_channels', 'out_channels', 'block_out_channels', 'layers_per_block')})
    dec.load_state_dict(lw['model_state_dict'])
    dec = dec.to(device, dtype=torch.float16).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    venc = load_venc(A, device)

    # 损失 (可微!)
    lpips_m = pyiqa.create_metric('lpips', device=device)
    lpips_m.as_loss = True
    clipiqa_m = pyiqa.create_metric('clipiqa', device=device)
    clipiqa_m.as_loss = True

    # 数据
    imgs = []
    for d in DATA_DIRS:
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            imgs += sorted(Path(d).glob(ext))
    if len(imgs) == 0:
        log(f'[fatal] 数据目录无图: {DATA_DIRS}')
        sys.exit(2)
    log(f'[data] {len(imgs)} 张 HQ 图')
    profile = EmpProfile(EMP_DEG, EMP_JPEG)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    # --- 全状态续训: 恢复 opt/sched/RNG, cosine 轨迹无缝 ---
    start_step = 0
    if args.resume_from:
        start_step = int(prev.get('step', 0))
        if int(prev.get('steps', args.steps)) != args.steps:
            log(f'[warn] T_max mismatch: 存档 steps={prev.get("steps")} != 本次 --steps {args.steps}')
        if 'opt_state' in prev:
            opt.load_state_dict(prev['opt_state'])
            sched.load_state_dict(prev['sched_state'])
            log(f'[resume] opt/sched restored (step {start_step}, lr={sched.get_last_lr()[0]:.2e})')
        else:
            # 旧格式 (精修存档): 只有权重, schedule 重建 (lr 从 3e-5 重来, 仅适用于精修场景)
            log('[warn] 旧格式存档 (无 opt/sched): schedule 重建, lr 从头 (精修语义)')
        if 'rng' in prev:
            torch.set_rng_state(torch.ByteTensor(prev['rng']['torch']))
            torch.cuda.set_rng_state(torch.ByteTensor(prev['rng']['cuda']))
            np.random.set_state(prev['rng']['np'])
            random.setstate(prev['rng']['py'])
            log('[resume] rng restored')

    t0 = time.time()

    def save_full(step_done, fname):
        """全状态存档: LoRA 权重 + opt/sched + RNG (无缝续训用)"""
        cfg_json = json.dumps({
            'r': lora_cfg.r, 'lora_alpha': lora_cfg.lora_alpha,
            'lora_dropout': lora_cfg.lora_dropout, 'bias': lora_cfg.bias,
            'target_modules': LORA_TARGETS,
        })
        ck = {
            'variant': args.variant, 'step': step_done, 'steps': args.steps,
            'lora_config': cfg_json,
            'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items() if 'lora' in k},
            'opt_state': {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                          for k, v in opt.state_dict().items()},
            'sched_state': sched.state_dict(),
            'rng': {
                'torch': torch.get_rng_state().tolist(),
                'cuda': torch.cuda.get_rng_state().tolist(),
                'np': np.random.get_state(),
                'py': random.getstate(),
            },
        }
        torch.save(ck, str(out_dir / fname))

    for step in range(start_step, args.steps):
        if step % 25 == 0 and PAUSE_FLAG.exists():
            save_full(step, 'lora_pause.pt')
            log('[pause] flag found, saved lora_pause.pt, exit (resume: --resume-from lora_pause.pt)')
            sys.exit(0)
        # 采样裁剪 + 退化
        ip = random.choice(imgs)
        im = Image.open(ip).convert('RGB')
        if im.width < args.patch or im.height < args.patch:
            im = im.resize((max(args.patch, im.width), max(args.patch, im.height)), Image.LANCZOS)
        x0 = random.randrange(im.width - args.patch + 1)
        y0 = random.randrange(im.height - args.patch + 1)
        gt = im.crop((x0, y0, x0 + args.patch, y0 + args.patch))
        lq = profile.degrade(gt)
        gt_t = torch.from_numpy(np.asarray(gt).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
        lq_t = torch.from_numpy(np.asarray(lq).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            with torch.no_grad():
                z_fea = venc_features(venc, lq_t, A)
                z_fea = [z.detach() for z in z_fea]
                lq_lat = vae.encode(lq_t.half()).latent_dist.sample() * vae.config.scaling_factor
            z0 = torch.randn_like(lq_lat)
            u = model(torch.cat([lq_lat, z0], dim=1),
                      torch.ones(lq_t.shape[0], device=device),
                      torch.zeros(lq_t.shape[0], device=device),
                      z_fea)
            z1 = z0 - u
            x1 = dec(z1 / vae.config.scaling_factor).clamp(-1, 1)

        x1_01 = ((x1.float() + 1) / 2).clamp(1e-4, 1 - 1e-4)
        gt_01 = ((gt_t.float() + 1) / 2).clamp(1e-4, 1 - 1e-4)
        loss_lp = lpips_m(x1_01, gt_01)
        loss_cq = (1 - clipiqa_m(x1_01)).mean()
        loss_l1 = F.l1_loss(x1_01, gt_01)
        loss = args.w_lpips * loss_lp + args.w_clip * loss_cq + args.w_l1 * loss_l1

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()

        if (step + 1) % 50 == 0:
            el = (time.time() - t0) / (step + 1)
            log(f'[step {step + 1}] loss={loss.item():.4f} lp={loss_lp.item():.4f} '
                f'cq={loss_cq.item():.4f} l1={loss_l1.item():.4f} lr={sched.get_last_lr()[0]:.2e} {el:.2f}s/step')
        if (step + 1) % args.save_every == 0:
            save_full(step + 1, f'{args.ckpt_prefix}_s{step + 1}.pt')
            log(f'[save] {args.ckpt_prefix}_s{step + 1}.pt')

    log('[done]')


if __name__ == '__main__':
    main()
