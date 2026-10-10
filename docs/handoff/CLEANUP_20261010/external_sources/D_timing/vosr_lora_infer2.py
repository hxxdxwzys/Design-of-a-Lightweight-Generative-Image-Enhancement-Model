# -*- coding: utf-8 -*-
"""VOSR LoRA 4K 渲染 (2026-08-18): 与 inference_vosr_onestep 同管线 + LoRA 权重注入
用法 (cwd=任意的):
  D:/Python/python.exe D:/tmp_timing/vosr_lora_infer.py -c C:/models/finals/preset/ckpts/VOSR_0.5B_os \
      --lora D:/models/vosr_lora/v1/lora_s3000.pt -i <lq_dir> -o <out_dir> \
      [--seed N] [--tile_size 512] [--vae_tile_size 1024] [--infer_steps 1] [--align_method wavelet|adain|nofix]
"""
import argparse
import json
import os
import sys
import time

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'  # Windows 无 triton: torch.compile 静默回退 eager

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

VOSR_DIR = r'D:/models/finals/VOSR'
sys.path.insert(0, VOSR_DIR)
sys.path.insert(0, r'D:/tmp_timing')

from peft import LoraConfig, get_peft_model  # noqa: E402
from models.lightningdit import LightningDiT  # noqa: E402
from diffusers import AutoencoderKL  # noqa: E402
from models.light_decoder import LightDecoder  # noqa: E402
from inference_vosr_onestep import (  # noqa: E402
    load_config_with_cli, load_dinov2, get_venc_features, tiled_latent_inference,
    adain_color_fix, wavelet_color_fix, list_lq_images, _inference_accelerator_stub,
)


# ---- 离线 dinov2 补丁 (2026-09-15 夜) ----
# 原 _hub_load_offline_first 用相对路径 'preset/ckpts/torch_cache', 只有 CWD=C:/models/finals
# 才命中本地仓库, 否则回落 torch.hub.load('facebookresearch/dinov2') 走网络 -> 断网 60s 超时崩。
import inference_vosr_onestep as _iv
os.environ.setdefault('TORCH_HOME', r'C:/models/finals/preset/ckpts/torch_cache')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
_HUB_DIR = r'C:/models/finals/preset/ckpts/torch_cache'


def _offline_first_abs(name):
    local = os.path.join(_HUB_DIR, 'facebookresearch_dinov2_main')
    if os.path.isdir(local):
        try:
            return torch.hub.load(local, name, source='local')
        except Exception as e:
            print('[warn] local dinov2 load failed (%s); fallback remote' % e, flush=True)
    return torch.hub.load('facebookresearch/dinov2', name)


_iv._hub_load_offline_first = _offline_first_abs
# ---- 补丁结束 ----

TORCH_CACHE = r'C:/models/finals/preset/ckpts/torch_cache'
LWDEC = r'C:/models/finals/preset/ckpts/sd21_lwdecoder.pth'


def build_model(args, lora_path, lora_scale=1.0):
    torch.hub.set_dir(TORCH_CACHE)
    base_channel = 4 if args.ae_type == 'sd2' else 16
    model = LightningDiT(
        input_size=args.resolution // 8,
        patch_size=args.patch_size,
        in_channels=2 * base_channel,
        out_channels=base_channel,
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
    weight_path = os.path.join(args.checkpoint, 'checkpoints', 'ema_model.safetensors')
    if not os.path.isfile(weight_path):
        weight_path = os.path.join(args.checkpoint, 'clean_weights', 'ema_model.safetensors')
    sd = load_file(weight_path)
    model.load_state_dict(sd, strict=False)
    model.forward = model.forward_flexible
    lora_cfg = json.loads(lora_path['lora_config'])
    model = get_peft_model(model, LoraConfig(**lora_cfg))
    lora_sd = {k: v for k, v in lora_path['state_dict'].items() if 'lora' in k}
    if lora_scale != 1.0:
        # 缩放 lora_B: W' = W + scale * (alpha/r) * B @ A
        lora_sd = {k: (v * lora_scale if 'lora_B' in k else v) for k, v in lora_sd.items()}
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    print(f'[lora] loaded {len(lora_sd)} keys (scale={lora_scale}), missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--lora', type=str, required=True)
    parser.add_argument('--lora-scale', type=float, default=1.0,
                        help='LoRA 强度插值系数 (0=基础模型, 1=原始 LoRA); 扫前沿用')
    parser.add_argument('--input_image', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--upscale', '-u', type=int, default=1)
    parser.add_argument('--align_method', choices=['wavelet', 'adain', 'nofix'], default='wavelet')
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--vae_tile_size', type=int, default=1024)
    parser.add_argument('--vae_tile_overlap', type=int, default=None)
    parser.add_argument('--infer_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='')
    temp_args, _ = parser.parse_known_args()
    args = load_config_with_cli(temp_args.checkpoint, parser)
    if getattr(args, 'auxiliary_time_cond', None) is None:
        args.auxiliary_time_cond = True
    args.auxiliary_time_cond = bool(args.auxiliary_time_cond)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = 'cuda'
    lora_path = torch.load(temp_args.lora, map_location='cpu', weights_only=False)
    print(f'[lora] {temp_args.lora}: variant={lora_path.get("variant")} step={lora_path.get("step")}', flush=True)
    model = build_model(args, lora_path, lora_scale=temp_args.lora_scale).to(device).eval()

    ae_path = os.path.join(os.path.dirname(args.checkpoint), 'stable-diffusion-2-1-base')
    vae = AutoencoderKL.from_pretrained(ae_path, subfolder='vae').to(device).eval()
    lw = torch.load(LWDEC, map_location='cpu', weights_only=False)
    dec = LightDecoder(**{k: lw['config'][k] for k in ('in_channels', 'out_channels', 'block_out_channels', 'layers_per_block')})
    dec.load_state_dict(lw['model_state_dict'])
    dec = dec.to(device).eval()
    venc = load_dinov2(args, device)

    image_paths = list_lq_images(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)
    to_tensor = transforms.ToTensor()
    print(f'[start] {len(image_paths)} images', flush=True)
    t0 = time.time()
    for idx, img_path in enumerate(image_paths):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        t_img = time.time()
        raw_img = Image.open(img_path).convert('RGB')
        w, h = raw_img.size
        target_w, target_h = w * args.upscale, h * args.upscale
        input_img = raw_img.resize((target_w, target_h), Image.BICUBIC)
        lq = to_tensor(input_img).unsqueeze(0).to(device) * 2.0 - 1.0
        with torch.no_grad():
            if args.tile_size > 0:
                sr = tiled_latent_inference(model, vae, venc, lq, args, device=device, light_decoder=dec)
            else:
                from tiled_vae import encode_dispatch, decode_dispatch
                lq_latent, mean, std = encode_dispatch(vae, lq, args, device)
                z_fea = get_venc_features(venc, lq, args)
                z = torch.randn_like(lq_latent)
                u = model(torch.cat([lq_latent, z], 1), torch.ones(lq.shape[0], device=device),
                          torch.zeros(lq.shape[0], device=device), z_fea)
                sr = decode_dispatch(vae, z - u, args, mean, std, light_decoder=dec)
        sr_img = transforms.ToPILImage()(sr[0].cpu() * 0.5 + 0.5)
        if args.align_method == 'adain':
            sr_img = adain_color_fix(sr_img, input_img)
        elif args.align_method == 'wavelet':
            sr_img = wavelet_color_fix(sr_img, input_img)
        sr_img.save(os.path.join(args.output_dir, stem + '.png'))
        print(f'[{idx + 1}/{len(image_paths)}] {stem} {time.time() - t_img:.0f}s', flush=True)
    print(f'[done] {args.output_dir} ({(time.time() - t0) / 60:.1f}min)', flush=True)


if __name__ == '__main__':
    main()
