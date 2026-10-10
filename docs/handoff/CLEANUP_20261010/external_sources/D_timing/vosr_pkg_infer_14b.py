# -*- coding: utf-8 -*-
"""VOSR 1.4B 合规 pkg 4K 渲染 (2026-09-10)

与 vosr_lora_infer_14b.py 同构, 唯一区别: 权重来源是**已合并的合规 pkg** (your_model.pt),
而不是"基座 + PEFT LoRA"。用于 (a) 复现提交包出图, (b) 渲染 remerge_scale_14b.py 产出的
任意 scale 档。

用法 (cwd=任意):
  D:/Python/python.exe D:/tmp_timing/vosr_pkg_infer_14b.py -c C:/models/finals/preset/ckpts/VOSR_1.4B_os \
      --pkg <your_model.pt> -i <lq_dir> -o <out_dir> --fp16 --align_method wavelet
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, r'D:/tmp_timing')

# ⚠️ 必须 chdir: onestep 的 _hub_load_offline_first() 用【相对路径】preset/ckpts/torch_cache
# 判定本地 dinov2 缓存是否可用。cwd 不对 -> 离线分支失效 -> 回落到 torch.hub 远程请求
# api.github.com (校园网下间歇性 WinError 10060 超时, 2026-09-11 实测 2/5 次渲染因此整轮失败)。
os.chdir(r'C:/models/finals')

from models.lightningdit import LightningDiT  # noqa: E402
from inference_vosr_onestep import (  # noqa: E402
    load_config_with_cli, load_dinov2, get_venc_features, tiled_latent_inference,
    adain_color_fix, wavelet_color_fix, list_lq_images,
)

TORCH_CACHE = r'C:/models/finals/preset/ckpts/torch_cache'
QWEN_AE = r'C:/models/finals/preset/ckpts/Qwen-Image-vae-2d'
SD2_AE = r'C:/models/finals/preset/ckpts/stable-diffusion-2-1-base'
LWDEC = r'C:/models/finals/preset/ckpts/sd21_lwdecoder.pth'


def build_from_pkg(args, pkg_path):
    """用 pkg['diT_cfg'] 建架构 + pkg['diT'] 覆盖权重 (不用 peft)。"""
    torch.hub.set_dir(TORCH_CACHE)
    pkg = torch.load(pkg_path, map_location='cpu', mmap=True, weights_only=False)
    cfg = dict(pkg['diT_cfg'])
    cfg.pop('lora_scale', None)
    model = LightningDiT(**cfg)
    model.forward = model.forward_flexible
    dt = pkg['diT']
    cur = model.state_dict()
    hit = 0
    for k, v in dt.items():
        if k in cur and cur[k].is_floating_point() and v.is_floating_point():
            if tuple(cur[k].shape) != tuple(v.shape):
                raise RuntimeError(f'形状不符 {k}: pkg={tuple(v.shape)} model={tuple(cur[k].shape)}')
            cur[k].copy_(v.float())
            hit += 1
    print(f'[pkg] {os.path.basename(pkg_path)}: lora_scale={pkg.get("lora_scale")} '
          f'注入 {hit}/{len(cur)} 键, 未命中={len(set(cur) - set(dt))}', flush=True)
    if hit == 0:
        raise RuntimeError('pkg diT 与模型 state_dict 无交集 — 架构不匹配')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--pkg', type=str, required=True)
    parser.add_argument('--input_image', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--upscale', '-u', type=int, default=1)
    parser.add_argument('--align_method', choices=['wavelet', 'adain', 'nofix'], default='wavelet')
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--vae_tile_size', type=int, default=1024)
    parser.add_argument('--vae_tile_overlap', type=int, default=None)
    parser.add_argument('--infer_steps', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
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
    model = build_from_pkg(args, temp_args.pkg).to(device).eval()
    if temp_args.fp16:
        model = model.half()

    if args.ae_type == 'qwen':
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        vae = AutoencoderKLQwenImage2D.from_pretrained(QWEN_AE).to(device).eval()
        dec = None
    else:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(SD2_AE, subfolder='vae').to(device).eval()
        lw = torch.load(LWDEC, map_location='cpu')
        from models.light_decoder import LightDecoder
        dec = LightDecoder(**{k: lw['config'][k] for k in
                              ('in_channels', 'out_channels', 'block_out_channels', 'layers_per_block')})
        dec.load_state_dict(lw['model_state_dict'])
        dec = dec.to(device).eval()
    venc = load_dinov2(args, device)

    image_paths = list_lq_images(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)
    to_tensor = transforms.ToTensor()
    print(f'[start] {len(image_paths)} images, ae={args.ae_type}', flush=True)
    t0 = time.time()
    for idx, img_path in enumerate(image_paths):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        t_img = time.time()
        raw_img = Image.open(img_path).convert('RGB')
        w, h = raw_img.size
        input_img = raw_img.resize((w * args.upscale, h * args.upscale), Image.BICUBIC)
        lq = to_tensor(input_img).unsqueeze(0).to(device) * 2.0 - 1.0
        with torch.no_grad():
            ctx = torch.autocast('cuda', dtype=torch.float16) if temp_args.fp16 else torch.no_grad()
            with ctx:
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
        sr_img = transforms.ToPILImage()(sr[0].float().cpu() * 0.5 + 0.5)
        if args.align_method == 'adain':
            sr_img = adain_color_fix(sr_img, input_img)
        elif args.align_method == 'wavelet':
            sr_img = wavelet_color_fix(sr_img, input_img)
        sr_img.save(os.path.join(args.output_dir, stem + '.png'))
        print(f'[{idx + 1}/{len(image_paths)}] {stem} {time.time() - t_img:.0f}s', flush=True)
    print(f'[done] {args.output_dir} ({(time.time() - t0) / 60:.1f}min)', flush=True)


if __name__ == '__main__':
    main()
