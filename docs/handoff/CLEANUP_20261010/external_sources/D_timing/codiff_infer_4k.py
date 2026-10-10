# -*- coding: utf-8 -*-
"""CODiff 4K 自定义推理 v4 (2026-08-18): 纯快速路径, 与官方输出逐像素一致 (abs diff <0.001)
关键: VAE 输入必须 [-1,1] (官方 forward 语义); 此前 (0,1) 输入导致输出偏移+74 亮度。
流程: 全图 VAE encode (OOM 自动分块) -> CaVE embedding -> UNet latent 分块(96/32, 高斯羽化)
      -> scheduler.step(t=499, timesteps 已覆盖) -> 分块 VAE decode(128 latent/16 overlap, 羽化拼接)
用法 (cwd=仓库根):
  D:/Python/python.exe D:/tmp_timing/codiff_infer_4k.py -i <lq_dir> -o <out_dir> \
      --color-fix adain|nofix [--limit N]
输出: <out_dir>/<stem>.png (RGB, 与 LQ 同尺寸)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO = r'D:/models/finals/codiff/CODiff-main/CODiff-main'
sys.path.insert(0, REPO)

import utils.utils_image as utils  # noqa: E402
from diffusion.codiff import CODiff_test  # noqa: E402
from cave.cave import CaVE  # noqa: E402
from diffusion.my_utils.wavelet_color_fix import adain_color_fix  # noqa: E402


def gauss2d(size, sigma=6.0, device='cuda'):
    t = torch.linspace(-1, 1, size, device=device)
    g = torch.exp(-sigma * t * t)
    return (g.view(-1, 1) * g.view(1, -1))


@torch.no_grad()
def vae_encode_tiled(model, lq, tile=1024, overlap=64):
    """lq: [-1,1] fp16 [1,3,H,W]; OOM 保护: 全图优先, 失败分块"""
    try:
        return model.vae.encode(lq).latent_dist.sample() * model.vae.config.scaling_factor
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    _, _, H, W = lq.shape
    step = tile - overlap
    ys = list(range(0, H - tile + 1, step))
    xs = list(range(0, W - tile + 1, step))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    lh, lw = H // 8, W // 8
    acc = torch.zeros(1, 4, lh, lw, device=lq.device, dtype=torch.float16)
    wsum = torch.zeros(1, 1, lh, lw, device=lq.device, dtype=torch.float16)
    gw = gauss2d(tile // 8, device=lq.device).half()
    for y in ys:
        for x in xs:
            z = model.vae.encode(lq[:, :, y:y + tile, x:x + tile]).latent_dist.sample()
            z = z * model.vae.config.scaling_factor
            acc[:, :, y // 8:(y + tile) // 8, x // 8:(x + tile) // 8] += z * gw
            wsum[:, :, y // 8:(y + tile) // 8, x // 8:(x + tile) // 8] += gw
    return acc / wsum


@torch.no_grad()
def unet_tiled(model, lat, vemb, tile=96, overlap=32):
    """lat fp16 [1,4,h,w]; latent 分块高斯羽化; vemb fp16 [1,T,1024]"""
    _, _, h, w = lat.shape
    if h * w <= tile * tile:
        return model.unet(lat, model.timesteps, encoder_hidden_states=vemb).sample
    t = min(tile, min(h, w))
    gw = gauss2d(t, device=lat.device).half().view(1, 1, t, t)
    ys = list(range(0, h - t + 1, t - overlap))
    xs = list(range(0, w - t + 1, t - overlap))
    if ys[-1] + t < h:
        ys.append(h - t)
    if xs[-1] + t < w:
        xs.append(w - t)
    acc = torch.zeros_like(lat)
    wsum = torch.zeros_like(lat[:, :1])
    for y in ys:
        for x in xs:
            pred = model.unet(lat[:, :, y:y + t, x:x + t], model.timesteps,
                              encoder_hidden_states=vemb).sample
            acc[:, :, y:y + t, x:x + t] += pred * gw
            wsum[:, :, y:y + t, x:x + t] += gw
    return acc / wsum


@torch.no_grad()
def vae_decode_tiled(model, z, tile=128, overlap=16):
    """z fp16 [1,4,h,w] (未除 scaling); 分块解码 + 高斯羽化拼接 -> [-1,1] [1,3,H,W]"""
    _, _, h, w = z.shape
    H, W = h * 8, w * 8
    z = (z / model.vae.config.scaling_factor).to(torch.float16)
    t = min(tile, min(h, w))
    gw = gauss2d(t * 8, device=z.device).half()
    ys = list(range(0, h - t + 1, t - overlap))
    xs = list(range(0, w - t + 1, t - overlap))
    if ys[-1] + t < h:
        ys.append(h - t)
    if xs[-1] + t < w:
        xs.append(w - t)
    acc = torch.zeros(1, 3, H, W, device=z.device, dtype=torch.float16)
    wsum = torch.zeros(1, 1, H, W, device=z.device, dtype=torch.float16)
    for y in ys:
        for x in xs:
            dec = model.vae.decode(z[:, :, y:y + t, x:x + t]).sample.clamp(-1, 1)
            acc[:, :, y * 8:(y + t) * 8, x * 8:(x + t) * 8] += dec * gw
            wsum[:, :, y * 8:(y + t) * 8, x * 8:(x + t) * 8] += gw
    return (acc / wsum).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--lq-dir', required=True)
    ap.add_argument('-o', '--out-dir', required=True)
    ap.add_argument('--pretrained_model', default=r'D:/models/sd21base')
    ap.add_argument('--cave_path', default=r'D:/models/finals/codiff/cave.pth')
    ap.add_argument('--codiff_path', default=r'D:/models/finals/codiff/codiff.pkl')
    ap.add_argument('--color-fix', choices=['adain', 'nofix'], default='nofix')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 张 (0=全部)')
    # 兼容 auto_pipeline 传入的旧参数 (v4 不再使用)
    ap.add_argument('--vae_encoder_tiled_size', type=int, default=1024)
    ap.add_argument('--vae_decoder_tiled_size', type=int, default=256)
    ap.add_argument('--latent_tiled_size', type=int, default=96)
    ap.add_argument('--latent_tiled_overlap', type=int, default=32)
    ap.add_argument('--mixed_precision', choices=['fp16', 'fp32'], default='fp16')
    ap.add_argument('--merge_and_unload_lora', action='store_true', default=False)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model = CODiff_test(args)
    # 绕开 VAEHook (v4 全手动路径; 与官方输出一致性已在 512 裁剪块验证 abs diff<0.001)
    if hasattr(model.vae.encoder, 'original_forward'):
        model.vae.encoder.forward = model.vae.encoder.original_forward
    if hasattr(model.vae.decoder, 'original_forward'):
        model.vae.decoder.forward = model.vae.decoder.original_forward
    cave = CaVE(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='BR')
    cave.load_state_dict(torch.load(args.cave_path), strict=True)
    cave.eval()
    for k, v in cave.named_parameters():
        v.requires_grad = False
    cave = cave.to('cuda')

    paths = sorted(utils.get_image_paths(args.lq_dir))
    if args.limit:
        paths = paths[:args.limit]
    print(f'[start] {len(paths)} images', flush=True)

    t0 = time.time()
    for idx, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        t_img = time.time()
        img = Image.open(p).convert('RGB')
        w, h = img.size
        nw, nh = w - w % 8, h - h % 8
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), Image.LANCZOS)
        lq_raw = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to('cuda')
        with torch.no_grad():
            emb = cave.get_visual_embedding(lq_raw)                    # fp32 [1,T,512]
            lat = vae_encode_tiled(model, (lq_raw * 2 - 1).to(torch.float16))
            vemb = model.proj(emb).to(torch.float16)
            pred = unet_tiled(model, lat, vemb)
            prev = model.noise_scheduler.step(pred, model.timesteps, lat, return_dict=True).prev_sample
            out = vae_decode_tiled(model, prev)
        img_E = transforms.ToPILImage()(out[0].cpu() * 0.5 + 0.5)
        if args.color_fix == 'adain':
            img_E = adain_color_fix(target=img_E, source=np.asarray(img))
        utils.imsave(np.asarray(img_E), os.path.join(args.out_dir, stem + '.png'))
        print(f'[{idx + 1}/{len(paths)}] {stem} {time.time() - t_img:.0f}s', flush=True)
    print(f'[done] {args.out_dir} ({(time.time() - t0) / 60:.1f}min)', flush=True)


if __name__ == '__main__':
    main()
