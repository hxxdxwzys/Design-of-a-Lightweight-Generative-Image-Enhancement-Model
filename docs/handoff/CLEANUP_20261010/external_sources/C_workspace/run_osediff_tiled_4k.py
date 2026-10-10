#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""官方 OSEDiff 分块 4K 推理 (绕开官方 test 的整图 attention OOM)。

思路: 用官方 venv (diffusers 0.25) 加载 OSEDiff_test, 把 4K 图切成 512 tile
逐块喂给 model.forward(lq_tile, prompt), 高斯融合拼接 (同 v3 infer.py)。
每块 tile 512x512, attention 在块内计算, 8GB 无压力。

用法 (官方 venv python 运行):
  D:/venv_osediff311/Scripts/python.exe run_osediff_tiled_4k.py <input_dir> <output_dir>
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, 'D:/models/OSEDiff-main/OSEDiff-main')
from osediff import OSEDiff_test


def gaussian_weights(tile, overlap):
    w = np.ones((tile, tile), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap + 2)[1:-1]
        for i in range(overlap):
            v = ramp[i]
            w[i, :] = np.minimum(w[i, :], v)
            w[-1 - i, :] = np.minimum(w[-1 - i, :], v)
            w[:, i] = np.minimum(w[:, i], v)
            w[:, -1 - i] = np.minimum(w[:, -1 - i], v)
    return torch.from_numpy(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_dir')
    ap.add_argument('output_dir')
    ap.add_argument('--sd_path', default='D:/models/sd21base')
    ap.add_argument('--osediff_dir', default='D:/models/OSEDiff-main/OSEDiff-main')
    ap.add_argument('--ram_path', default='D:/models/ram_swin_large_14m.pth')
    ap.add_argument('--dape_path', default=None)
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=64)
    ap.add_argument('--prompt', default='', help='额外 prompt; 空则用固定通用 prompt')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载官方 OSEDiff (diffusers 0.25 环境)
    class _Args:
        pass
    a = _Args()
    a.pretrained_model_name_or_path = args.sd_path
    a.osediff_path = os.path.join(args.osediff_dir, 'preset/models/osediff.pkl')
    a.mixed_precision = 'fp16'
    a.merge_and_unload_lora = False
    a.latent_tiled_size = 96
    a.latent_tiled_overlap = 32
    a.vae_encoder_tiled_size = 512
    a.vae_decoder_tiled_size = 224
    model = OSEDiff_test(a)
    model.eval()

    # RAM 标签 (可选)
    ram = None
    if os.path.exists(args.ram_path):
        try:
            from ram.models.ram_lora import ram as ram_cls
            from ram import inference_ram as inference
            from ram import get_transform
            ram = ram_cls(pretrained=args.ram_path, pretrained_condition=args.dape_path,
                          image_size=384, vit='swin_l').to('cuda').eval().half()
            ram_tf = get_transform(image_size=384)
            print('[ram] loaded', flush=True)
        except Exception as e:
            print(f'[ram] skip: {e}', flush=True)
            ram = None

    def get_prompt(img_pil):
        if ram is not None:
            try:
                t = ram_tf(img_pil.convert('RGB')).unsqueeze(0).to('cuda').half()
                caps = inference(t, ram)
                if caps and caps[0]:
                    return f'{caps[0]}, {args.prompt},'
            except Exception:
                pass
        return 'a photo of a high-quality, sharp, clean scene,'

    files = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    for i, f in enumerate(files):
        img = Image.open(f).convert('RGB')
        W, H = img.size
        prompt = get_prompt(img)
        print(f'[{i+1}/{len(files)}] {os.path.basename(f)} {W}x{H} prompt={prompt[:40]}', flush=True)

        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        tile, ov = args.tile, args.overlap
        pw = (tile - W % tile) % tile
        ph = (tile - H % tile) % tile
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode='reflect')
        H2, W2 = arr.shape[:2]
        t_in = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to('cuda')
        out = torch.zeros(1, 3, H2, W2, device='cuda')
        wsum = torch.zeros(1, 1, H2, W2, device='cuda')
        g = gaussian_weights(tile, ov).to('cuda')
        step = tile - ov
        with torch.no_grad():
            for y in range(0, H2, step):
                for x in range(0, W2, step):
                    y1 = min(y, H2 - tile)
                    x1 = min(x, W2 - tile)
                    patch = t_in[:, :, y1:y1 + tile, x1:x1 + tile]
                    enh = model(patch, prompt).float()
                    out[:, :, y1:y1 + tile, x1:x1 + tile] += enh * g
                    wsum[:, :, y1:y1 + tile, x1:x1 + tile] += g
        out = out / wsum.clamp(min=1e-8)
        out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5)
        out_np = np.clip(out_np[:H, :W], 0, 255).astype(np.uint8)
        Image.fromarray(out_np).save(os.path.join(args.output_dir, os.path.basename(f).replace('.png', '.jpg')), quality=95)
        print(f'  -> saved {os.path.basename(f).replace(chr(46)+chr(112)+chr(110)+chr(103), chr(46)+chr(106)+chr(112)+chr(103))}', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
