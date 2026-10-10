# -*- coding: utf-8 -*-
"""FiDeSR 分块 4x 快速版 (2026-08-25): 进程内加载模型一次, 逐块推理 (val/测试共用)
用法: cd FiDeSR-main && D:/Python/python.exe D:/tmp_timing/b_fidesr_tiled4x_fast.py -i <dir> -o <out> [--lf 0.2] [--hf 0.3] [--align wavelet] [--tag tiled4x]
特性: 逐图幂等 + 暂停标记 (fidesr_pause.flag, 图像边界)
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
sys.path.insert(0, r'D:/models/new_src/fidesr/code/FiDeSR-main')
os.chdir(r'D:/models/new_src/fidesr/code/FiDeSR-main')

import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace

TT = Path(r'D:/tmp_timing')
PAUSE = TT / 'fidesr_pause.flag'
TILE = 1024
# UP = 4  (由 --up 覆盖)


def log(msg):
    print(msg, flush=True)
    with (TT / 'b_fidesr_tiled4x_fast.log').open('a', encoding='utf-8') as f:
        f.write(f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}\n')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--lf', type=float, default=0.2)
    ap.add_argument('--hf', type=float, default=0.3)
    ap.add_argument('--align', default='wavelet')
    ap.add_argument('--up', type=int, default=4, help='2 或 4 (倍率)')
    args = ap.parse_args()

    from fidesr import FiDeSR_eval
    from src.my_utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

    UP = args.up
    m_args = SimpleNamespace(
        pretrained_model_path=r'C:/models/finals/preset/ckpts/stable-diffusion-2-1-base',
        pretrained_path=r'D:/models/new_src/fidesr/fidesr.pkl',
        lrrb_in_ch=8, lrrb_mid_ch=64, lrrb_growth=32, lrrb_res_sc=0.2,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,  # 解码注意力限制: 224 是最大安全块 (1107任务 ~6min)
        latent_tiled_size=96, latent_tiled_overlap=32,
        mixed_precision='fp16',
        lf_scale=args.lf, hf_scale=args.hf,
        lf_rc=0.30, lf_order=2, lf_tau=0.8, lf_sharp=10.0, lf_dmap_gamma=1.2,
        hf_rc=0.32, hf_order=2, hf_dmap_gamma=1.2, lf_hf_merge_ratio=0.5,
    )
    log('[load] building FiDeSR_eval...')
    m = FiDeSR_eval(m_args)
    m.set_eval()
    log('[load] model ready')

    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    inp = Path(args.input)
    imgs = sorted(inp.glob('*.jpg')) + sorted(inp.glob('*.png')) if inp.is_dir() else [inp]
    imgs = list(dict.fromkeys(imgs))
    log(f'[boot] {len(imgs)} images, lf={args.lf} hf={args.hf} align={args.align}')

    for img_path in imgs:
        stem = img_path.stem
        outp = out_dir / f'{stem}.png'
        if outp.exists():
            log(f'[skip] {stem} exists')
            continue
        if PAUSE.exists():
            log('[pause] flag found, clean stop')
            sys.exit(0)
        t0 = time.time()
        im = Image.open(img_path).convert('RGB')
        W, H = im.size
        Wp = ((W + TILE - 1) // TILE) * TILE
        Hp = ((H + TILE - 1) // TILE) * TILE
        if (Wp, Hp) != (W, H):
            pad_im = Image.new('RGB', (Wp, Hp))
            pad_im.paste(im, (0, 0))
            im = pad_im
        canvas = Image.new('RGB', (Wp, Hp))
        n = 0
        for y0 in range(0, Hp, TILE):
            for x0 in range(0, Wp, TILE):
                tile = im.crop((x0, y0, x0 + TILE, y0 + TILE))
                # 4x 语义: 输入先 bicubic x4 (test_fidesr.py 同款), 输出再 4x 下采样
                tile4 = tile.resize((TILE * UP, TILE * UP), Image.BICUBIC)
                c_t = to_tensor(tile4).unsqueeze(0).cuda() * 2 - 1
                with torch.no_grad():
                    _, out_img = m(c_t, prompt='')
                out_img = out_img * 0.5 + 0.5
                out_img = torch.clip(out_img, 0, 1)
                out_pil = transforms.ToPILImage()(out_img[0].cpu())
                if args.align == 'wavelet':
                    out_pil = wavelet_color_fix(out_pil, tile4)  # 4x 源对齐 (尺寸须一致)
                elif args.align == 'adain':
                    out_pil = adain_color_fix(out_pil, tile)
                out1 = out_pil.resize((TILE, TILE), Image.LANCZOS)
                canvas.paste(out1, (x0, y0))
                n += 1
                log(f'  [{stem}] tile {n} done ({time.time()-t0:.0f}s)')
        canvas = canvas.crop((0, 0, W, H))
        canvas.save(outp)
        log(f'[done] {stem} ({n} tiles, {time.time()-t0:.0f}s)')
    log('[done] all')


if __name__ == '__main__':
    main()
