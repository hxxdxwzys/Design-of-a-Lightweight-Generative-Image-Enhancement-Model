# -*- coding: utf-8 -*-
"""VOSR 0.5B ms 多步 4K 渲染 + 可选 v7 LoRA 注入 (2026-08-24)
基于 D:/models/finals/VOSR/inference_vosr.py 的多步 CFG 管线, 增加 --lora/--lora-scale。
关键: cwd 必须为 C:/models/finals (脚本内相对路径 preset/ckpts/... 依赖 cwd)。
用法:
  D:/Python/python.exe D:/tmp_timing/vosr_ms_infer.py -c C:/models/finals/preset/ckpts/VOSR_0.5B_ms \
      -i <lq_dir> -o <out_dir> --upscale 1 --infer_steps 8 --cfg_scale 2.0 \
      [--lora D:/models/vosr_lora/v7/lora_s5000.pt] [--lora-scale 1.0] \
      --tile_size 512 --tile_overlap 32 --vae_tile_size 1024 --seed 42 [--force_rerun]
"""
import argparse
import gc
import json
import os
import sys
import time

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

VOSR_DIR = r'D:/models/finals/VOSR'
sys.path.insert(0, VOSR_DIR)

from inference_vosr import (  # noqa: E402
    load_config_with_cli, load_dinov2, tiled_latent_inference,
    get_venc_features, list_lq_images, all_expected_outputs_exist,
    wavelet_color_fix, adain_color_fix, _inference_accelerator_stub,
    SD2_LWDECODER_PATH, SD2_AE_PATH,
)
from vosr import VOSR  # noqa: E402
from tiled_vae import encode_dispatch, decode_dispatch  # noqa: E402
from models.lightningdit import LightningDiT  # noqa: E402
from models.light_decoder import LightDecoder  # noqa: E402
from diffusers import AutoencoderKL  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--input_image', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument("--upscale", '-u', type=int, default=1)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='wavelet')
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=32)
    parser.add_argument('--vae_tile_size', type=int, default=1024)
    parser.add_argument('--vae_tile_overlap', type=int, default=None)
    parser.add_argument('--posterior_mode', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--infer_steps', type=int, default=25)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--weak_cond_strength_aelq', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force_rerun', action='store_true')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--lora', type=str, default=None, help='LoRA ckpt 路径 (v7 等)')
    parser.add_argument('--lora-scale', type=float, default=1.0)

    temp_args, _ = parser.parse_known_args()
    args = load_config_with_cli(temp_args.checkpoint, parser)
    if args.infer_steps is None:
        args.infer_steps = 25

    image_paths = list_lq_images(args.input_image)
    if not image_paths:
        print("No LQ images found. Exit.")
        return
    out_dir = args.output_dir
    if (
        not getattr(args, 'force_rerun', False)
        and not os.environ.get('INFER_NO_SKIP', '').lower() in ('1', 'true', 'yes')
        and all_expected_outputs_exist(image_paths, [out_dir])
    ):
        print(f"Skip: all {len(image_paths)} outputs already exist. Use --force_rerun to override.")
        return

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = "cuda"
    accelerator = _inference_accelerator_stub()
    print(f"Loading model from {args.checkpoint}...", flush=True)

    vae = AutoencoderKL.from_pretrained(SD2_AE_PATH, subfolder="vae").to(device).eval()
    dec_ck = torch.load(SD2_LWDECODER_PATH, map_location="cpu")
    light_decoder = LightDecoder(
        in_channels=dec_ck["config"]["in_channels"],
        out_channels=dec_ck["config"]["out_channels"],
        block_out_channels=tuple(dec_ck["config"]["block_out_channels"]),
        layers_per_block=dec_ck["config"]["layers_per_block"],
    )
    light_decoder.load_state_dict(dec_ck["model_state_dict"])
    light_decoder.to(device).eval()
    venc = load_dinov2(args, device)

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
        auxiliary_time_cond=False,
        use_qknorm=args.use_qknorm,
        use_swiglu=args.use_swiglu,
        use_rope=args.use_rope,
        use_rmsnorm=args.use_rmsnorm,
        wo_shift=args.wo_shift,
        num_fused_layers=len(args.layer_dinov2b_list),
    )
    import glob as _glob
    search_dirs = [
        os.path.join(args.checkpoint, "clean_weights"),
        os.path.join(args.checkpoint, "checkpoints"),
        args.checkpoint,
    ]
    weight_path = None
    for d in search_dirs:
        for name in ("ema_model.safetensors", "model.safetensors"):
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
                weight_path = cand
                break
        if weight_path:
            break
    if weight_path is None:
        hits = _glob.glob(os.path.join(args.checkpoint, "**/*.safetensors"), recursive=True)
        if not hits:
            raise FileNotFoundError(f"No .safetensors found under {args.checkpoint}")
        weight_path = hits[0]
    from safetensors.torch import load_file
    print(f"Loading weights from {weight_path}", flush=True)
    sd = load_file(weight_path)
    model.load_state_dict(sd, strict=False)
    model.forward = model.forward_flexible

    if temp_args.lora:
        lora_path = torch.load(temp_args.lora, map_location='cpu')
        cfg = json.loads(lora_path['lora_config'])
        lora_cfg = LoraConfig(
            r=cfg['r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg.get('lora_dropout', 0.0),
            bias=cfg.get('bias', 'none'), target_modules=cfg['target_modules'],
        )
        model = get_peft_model(model, lora_cfg)
        lora_sd = {k: v for k, v in lora_path['state_dict'].items() if 'lora' in k}
        if temp_args.lora_scale != 1.0:
            lora_sd = {k: (v * temp_args.lora_scale if 'lora_B' in k else v) for k, v in lora_sd.items()}
        missing, unexpected = model.load_state_dict(lora_sd, strict=False)
        print(f"[lora] {temp_args.lora} scale={temp_args.lora_scale} "
              f"loaded={len(lora_sd)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device).eval()

    vosr_model = VOSR(
        time_dist=args.time_dist,
        cfg_ratio=args.cfg_ratio,
        cfg_scale=args.cfg_scale,
        interp_type=args.interp_type,
        accelerator=accelerator,
        t_start=getattr(args, 't_start', 0.0),
        t_end=getattr(args, 't_end', 1.0),
        args=args,
    )

    os.makedirs(out_dir, exist_ok=True)
    to_tensor = transforms.ToTensor()
    print(f"Processing {len(image_paths)} images... steps={args.infer_steps} cfg={args.cfg_scale}", flush=True)
    t0 = time.time()
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        raw_img = Image.open(img_path).convert("RGB")
        w, h = raw_img.size
        target_w, target_h = w * args.upscale, h * args.upscale
        input_img = raw_img.resize((target_w, target_h), Image.BICUBIC)
        lq = to_tensor(input_img).unsqueeze(0).to(device) * 2.0 - 1.0
        t_img = time.time()
        with torch.no_grad():
            if args.tile_size > 0:
                sr_tensor = tiled_latent_inference(
                    model, vosr_model, vae, venc, lq, args,
                    device=device, light_decoder=light_decoder,
                )
            else:
                lq_latent, latents_mean, latents_std = encode_dispatch(vae, lq, args, device)
                z_fea = get_venc_features(venc, lq, args)
                sr_latent = vosr_model.sample_multistep_fm(
                    model, lq_latent, n_steps=args.infer_steps, venc_fea=z_fea
                )
                sr_tensor = decode_dispatch(vae, sr_latent, args, latents_mean, latents_std, light_decoder)
        sr_img = transforms.ToPILImage()(sr_tensor[0].cpu() * 0.5 + 0.5)
        if args.align_method == 'adain':
            sr_img = adain_color_fix(sr_img, input_img)
        elif args.align_method == 'wavelet':
            sr_img = wavelet_color_fix(sr_img, input_img)
        sr_img.save(os.path.join(out_dir, os.path.splitext(img_name)[0] + '.png'))
        print(f"[done] {img_name} ({time.time() - t_img:.0f}s)", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(f"Done! total {(time.time() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
