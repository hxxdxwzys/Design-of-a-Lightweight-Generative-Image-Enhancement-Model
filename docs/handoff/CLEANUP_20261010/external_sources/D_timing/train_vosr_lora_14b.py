# -*- coding: utf-8 -*-
"""B: VOSR 1.4B os LoRA 感知微调 (2026-08-24 初版 / 2026-08-31 Phase0 服务器版)

与 train_vosr_lora.py (0.5B os) 同构, 差异:
  - 基座 VOSR_1.4B_os: dim1536/depth36/qwen VAE 16ch/dinov2l
  - 混合精度: 基座+VAE+venc 全部 fp16 (冻结), LoRA fp32, autocast 前向 (可微损失在 fp32)
  - qwen 编解码: encode (z-mean)*(1/std); decode z/std+mean -> vae.decode

Phase0 (服务器) 改造:
  - --batch N: 批量训练 (默认 4; A100-40GB 建议 4, H100-80GB 可 8)
  - --accum N: 梯度累积 (默认 1; 40GB 上 batch2+accum2 亦可)
  - --patch 默认 1024 (NEXT_ROUND 决策); 本地 4060 冒烟必须 --patch 128/512
  - --save-every 默认 500
  - 路径全部参数化 (--out/--data-dirs/--ckpt/--qwen-ae/--torch-cache/--deg-json/--jpeg-json/--pause-flag/--vosr-dir/--tools-dir)
  - 暂停续跑协议保留 (v14_pause.flag 模式, 路径可配)

用法 (服务器):
  python train_vosr_lora_14b.py --variant v14p --w-clip 0 --steps 2000 --patch 1024 --batch 4 \
      --data-dirs /data/datasets/div2k/pngtxt_ori /data/datasets/flickr2k \
      --out /data/proj/runs/vosr_lora --ckpt /data/weights/VOSR_1.4B_os \
      --qwen-ae /data/weights/Qwen-Image-vae-2d --pause-flag /data/proj/v14_pause.flag \
      --vosr-dir /data/proj/VOSR --tools-dir /data/proj/tools
  续训: 加 --resume-from <ckpt> (--steps 必须与原始一致, 全状态恢复)
  暂停: touch 暂停标记文件 -> 每 25 步检查 -> 存 lora_pause.pt 干净退出

输出: {out}/{variant}/lora_s{step}.pt
"""
import argparse
import json
import math  # noqa: F401
import os
import random
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

LORA_TARGETS = ['qkv', 'proj', 'w12', 'w3', 'fc1', 'fc2', 'linear']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--w-clip', type=float, required=True)
    ap.add_argument('--w-lpips', type=float, default=1.0)
    ap.add_argument('--w-l1', type=float, default=0.05)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--alpha', type=int, default=64)
    ap.add_argument('--patch', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--save-every', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume-from', default=None)
    ap.add_argument('--ckpt-prefix', default='lora')
    # 路径参数化 (服务器与本地不同)
    ap.add_argument('--out', default=r'D:/models/vosr_lora')
    ap.add_argument('--data-dirs', nargs='+',
                    default=[r'D:/datasets/div2k/pngtxt_ori', r'D:/datasets/flickr2k'])
    ap.add_argument('--ckpt', default=r'C:/models/finals/preset/ckpts/VOSR_1.4B_os')
    ap.add_argument('--qwen-ae', default=r'C:/models/finals/preset/ckpts/Qwen-Image-vae-2d')
    ap.add_argument('--torch-cache', default=r'C:/models/finals/preset/ckpts/torch_cache')
    ap.add_argument('--deg-json', default=r'D:/tmp_timing/degradation_analysis.json')
    ap.add_argument('--jpeg-json', default=r'D:/tmp_timing/jpeg_blockiness.json')
    ap.add_argument('--pause-flag', default=r'D:/tmp_timing/v14_pause.flag')
    ap.add_argument('--vosr-dir', default=r'D:/models/finals/VOSR')
    ap.add_argument('--tools-dir', default=r'D:/tmp_timing')
    args = ap.parse_args()

    sys.path.insert(0, args.vosr_dir)
    sys.path.insert(0, args.tools_dir)

    import pyiqa  # noqa: E402
    from peft import LoraConfig, get_peft_model  # noqa: E402
    from models.lightningdit import LightningDiT  # noqa: E402
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D  # noqa: E402
    from train_v2 import EmpProfile  # noqa: E402

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = 'cuda'

    out_dir = Path(args.out) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = out_dir / 'train.log'

    def log(msg):
        line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        with logf.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    def load_base_model():
        torch.hub.set_dir(args.torch_cache)
        base = json.load(open(os.path.join(args.ckpt, 'args.json')))
        base['auxiliary_time_cond'] = True
        A = argparse.Namespace(**base)
        base_channel = 4 if A.ae_type == 'sd2' else 16
        model = LightningDiT(
            input_size=A.resolution // 8,
            patch_size=A.patch_size,
            in_channels=2 * base_channel,
            out_channels=base_channel,
            hidden_size=A.dim,
            depth=A.depth,
            num_heads=A.num_heads,
            mlp_ratio=A.mlp_ratio,
            z_dims=A.enc_dim,
            encdim_ratio=A.encdim_ratio,
            auxiliary_time_cond=A.auxiliary_time_cond,
            use_qknorm=A.use_qknorm,
            use_swiglu=A.use_swiglu,
            use_rope=A.use_rope,
            use_rmsnorm=A.use_rmsnorm,
            wo_shift=A.wo_shift,
            num_fused_layers=len(A.layer_dinov2b_list),
        )
        wpath = Path(args.ckpt) / 'checkpoints' / 'ema_model.safetensors'
        sd = load_file(str(wpath))
        model.load_state_dict(sd, strict=False)
        model.forward = model.forward_flexible
        model.eval()
        return model, A

    def load_venc(A):
        torch.hub.set_dir(args.torch_cache)
        import types
        hub_name = 'dinov2_vitl14' if A.enc_type == 'dinov2l' else 'dinov2_vitb14'
        # 本机 GitHub 被墙: 优先本地 hub 缓存仓库 (source='local', 完全离线);
        # 权重 dinov2_vitl14_pretrain.pth 在 torch_cache/checkpoints 已缓存, 无网络依赖
        local_repo = os.path.join(args.torch_cache, 'facebookresearch_dinov2_main')
        if os.path.isdir(local_repo):
            try:
                encoder = torch.hub.load(local_repo, hub_name, source='local')
                log(f'[venc] {hub_name} loaded from local hub cache (offline)')
            except Exception as e:
                log(f'[warn] local dinov2 load failed ({e}); fallback remote torch.hub')
                encoder = torch.hub.load('facebookresearch/dinov2', hub_name)
        else:
            encoder = torch.hub.load('facebookresearch/dinov2', hub_name)
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
        return encoder, A

    @torch.no_grad()
    def venc_features(venc, lq_tensor, A):
        raw_image = (0.5 * lq_tensor + 0.5) * 255
        x = F.interpolate(raw_image, A.dinov2_size, mode='bicubic').clip(0., 255.) / 255.
        x = (x - torch.tensor(IMAGENET_DEFAULT_MEAN, device=x.device).view(1, 3, 1, 1)) / \
            torch.tensor(IMAGENET_DEFAULT_STD, device=x.device).view(1, 3, 1, 1)
        features, x_norm = venc.forward_with_features(x)
        z = [v for k, v in features.items() if k.startswith('layer_')]
        z[-1] = x_norm
        z = [z[i] for i in A.layer_dinov2b_list]
        return z

    # 模型 + LoRA (基座 fp16 冻结, LoRA fp32)
    model, A = load_base_model()
    model = model.to(device).half()
    if args.resume_from:
        prev = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        cfg = json.loads(prev['lora_config'])
        lora_cfg = LoraConfig(r=cfg['r'], lora_alpha=cfg['lora_alpha'],
                              lora_dropout=cfg.get('lora_dropout', 0.0),
                              bias=cfg.get('bias', 'none'), target_modules=cfg['target_modules'])
        log(f'[resume] {args.resume_from} (step={prev.get("step")})')
    else:
        lora_cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                              bias='none', target_modules=LORA_TARGETS)
    model = get_peft_model(model, lora_cfg)
    if args.resume_from:
        lora_sd = {k: v for k, v in prev['state_dict'].items() if 'lora' in k}
        missing, unexpected = model.load_state_dict(lora_sd, strict=False)
        log(f'[resume] loaded {len(lora_sd)} keys, missing={len(missing)} unexpected={len(unexpected)}')
    model.print_trainable_parameters()

    log(f'[boot] variant={args.variant} w_clip={args.w_clip} steps={args.steps} lr={args.lr} '
        f'rank={args.rank} patch={args.patch} batch={args.batch} accum={args.accum}')

    # qwen VAE (fp16 冻结)
    vae = AutoencoderKLQwenImage2D.from_pretrained(args.qwen_ae).to(device).half().eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    lat_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1).to(device)
    lat_std_inv = (1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)).to(device)

    venc, _ = load_venc(A)
    venc = venc.half()

    # 损失 (可微, fp32)
    lpips_m = pyiqa.create_metric('lpips', device=device)
    lpips_m.as_loss = True
    clipiqa_m = pyiqa.create_metric('clipiqa', device=device)
    clipiqa_m.as_loss = True

    # 数据
    imgs = []
    for d in args.data_dirs:
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            imgs += sorted(Path(d).glob(ext))
    if len(imgs) == 0:
        log(f'[fatal] 数据目录无图: {args.data_dirs}')
        sys.exit(2)
    log(f'[data] {len(imgs)} 张 HQ 图')
    profile = EmpProfile(args.deg_json, args.jpeg_json)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    # --- 全状态续训: 恢复 opt/sched/RNG, 从已保存步数继续 (LR 轨迹无缝) ---
    start_step = 0
    if args.resume_from:
        start_step = int(prev.get('step', 0))
        if int(prev.get('steps', args.steps)) != args.steps:
            log(f'[warn] T_max mismatch: 存档 steps={prev.get("steps")} != 本次 --steps {args.steps} (cosine 轨迹会变, 应保持一致)')
        if 'opt_state' in prev:
            opt.load_state_dict(prev['opt_state'])
            sched.load_state_dict(prev['sched_state'])
            log(f'[resume] opt/sched restored (step {start_step}, lr={sched.get_last_lr()[0]:.2e})')
        else:
            # 旧格式存档: 只有权重, 重建 schedule 对齐原轨迹 (lr 连续性近似)
            sched.last_epoch = max(start_step - 1, -1)
            sched._step_count = start_step
            log(f'[warn] 旧格式存档 (无 opt/sched): schedule 重建于 step {start_step} (lr 将按 cosine 原轨迹续)')
        if 'rng' in prev:
            torch.set_rng_state(torch.ByteTensor(prev['rng']['torch']))
            torch.cuda.set_rng_state(torch.ByteTensor(prev['rng']['cuda']))
            np.random.set_state(prev['rng']['np'])
            random.setstate(prev['rng']['py'])
            log(f'[resume] rng restored')

    t0 = time.time()

    def save_ckpt(step_done, fname):
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
        log(f'[save] {fname} (step {step_done})')

    def make_batch():
        """采样 batch 张 patch 并退化, 返回 [B,3,H,W] 的 gt/lq ([-1,1], fp32, cuda)。"""
        gts, lqs = [], []
        for _ in range(args.batch):
            ip = random.choice(imgs)
            im = Image.open(ip).convert('RGB')
            if im.width < args.patch or im.height < args.patch:
                im = im.resize((max(args.patch, im.width), max(args.patch, im.height)), Image.LANCZOS)
            x0 = random.randrange(im.width - args.patch + 1)
            y0 = random.randrange(im.height - args.patch + 1)
            gt = im.crop((x0, y0, x0 + args.patch, y0 + args.patch))
            lq = profile.degrade(gt)
            gts.append(np.asarray(gt).astype(np.float32) / 127.5 - 1.0)
            lqs.append(np.asarray(lq).astype(np.float32) / 127.5 - 1.0)
        gt_t = torch.from_numpy(np.stack(gts)).permute(0, 3, 1, 2).to(device)
        lq_t = torch.from_numpy(np.stack(lqs)).permute(0, 3, 1, 2).to(device)
        return gt_t, lq_t

    PAUSE_FLAG = Path(args.pause_flag)
    for step in range(start_step, args.steps):
        if step % 25 == 0 and PAUSE_FLAG.exists():
            save_ckpt(step, 'lora_pause.pt')
            log('[pause] flag found, saved lora_pause.pt, exit (resume: --resume-from lora_pause.pt)')
            sys.exit(0)

        opt.zero_grad(set_to_none=True)
        loss_acc = lp_acc = cq_acc = l1_acc = 0.0
        for mb in range(args.accum):
            gt_t, lq_t = make_batch()
            with torch.autocast('cuda', dtype=torch.float16):
                with torch.no_grad():
                    z_fea = venc_features(venc, lq_t, A)
                    z_fea = [z.detach() for z in z_fea]
                    lq_enc = vae.encode(lq_t.half()).latent_dist.sample()
                    lq_lat = (lq_enc - lat_mean) * lat_std_inv
                z0 = torch.randn_like(lq_lat)
                u = model(torch.cat([lq_lat, z0], dim=1),
                          torch.ones(lq_t.shape[0], device=device),
                          torch.zeros(lq_t.shape[0], device=device),
                          z_fea)
                z1 = z0 - u
                x1 = vae.decode(z1 / lat_std_inv + lat_mean, return_dict=False)[0].clamp(-1, 1)

            x1_01 = ((x1.float() + 1) / 2).clamp(1e-4, 1 - 1e-4)
            gt_01 = ((gt_t.float() + 1) / 2).clamp(1e-4, 1 - 1e-4)
            loss_lp = lpips_m(x1_01, gt_01)
            loss_cq = (1 - clipiqa_m(x1_01)).mean()
            loss_l1 = F.l1_loss(x1_01, gt_01)
            loss = args.w_lpips * loss_lp + args.w_clip * loss_cq + args.w_l1 * loss_l1
            (loss / args.accum).backward()
            loss_acc += loss.item() / args.accum
            lp_acc += loss_lp.item() / args.accum
            cq_acc += loss_cq.item() / args.accum
            l1_acc += loss_l1.item() / args.accum

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()

        if (step + 1) % 50 == 0:
            el = (time.time() - t0) / (step + 1)
            log(f'[step {step + 1}] loss={loss_acc:.4f} lp={lp_acc:.4f} '
                f'cq={cq_acc:.4f} l1={l1_acc:.4f} lr={sched.get_last_lr()[0]:.2e} {el:.2f}s/step')
        if (step + 1) % args.save_every == 0:
            save_ckpt(step + 1, f'{args.ckpt_prefix}_s{step + 1}.pt')

    log('[done]')


if __name__ == '__main__':
    main()
