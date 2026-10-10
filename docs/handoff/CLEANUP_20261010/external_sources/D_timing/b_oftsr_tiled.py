# -*- coding: utf-8 -*-
"""OFTSR 分块推理驱动 (2026-08-26): DIV2K 一步蒸馏, 512px 块 + 32px 羽化重叠
单步: x0 = perturb(cond); v = model(cat([x0,cond])); out = x0 + (T-eps)*v
特性: 逐图幂等 + 暂停标记 (D:/tmp_timing/fidesr_pause.flag)
用法: cd OFTSR-main && D:/Python/python.exe D:/tmp_timing/b_oftsr_tiled.py -i <dir|file> -o <out> [--up 1|2|4]
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r'D:/models/new_src/oftsr/code/OFTSR-main')
os.chdir(r'D:/models/new_src/oftsr/code/OFTSR-main')

import numpy as np
import torch
import yaml
from types import SimpleNamespace
from PIL import Image
from torchvision import transforms

TT = Path(r'D:/tmp_timing')
PAUSE = TT / 'fidesr_pause.flag'
CKPT = r'D:/models/new_src/oftsr/div2k_onestep.pth'
TILE = 512
OVERLAP = 32
SIGMA_P = 0.2
T_ONE = 0.99
EPS = 0.001


def log(msg):
    print(msg, flush=True)
    with (TT / 'b_oftsr_tiled.log').open('a', encoding='utf-8') as f:
        f.write(f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}\n')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--up', type=int, default=1, help='1/2/4: 输入先放大再还原, 输出下采样回原尺寸')
    args = ap.parse_args()

    from models import create_model
    cfg = yaml.safe_load(open('configs/dis_fm_DIV2K.yml'))
    net = SimpleNamespace(**cfg['network'])
    net.img_size = cfg['dataset']['img_size']
    net.num_classes = cfg['dataset']['num_classes']
    net.world_size = 1
    net.use_cond = True
    net.attention_resolutions = "0" if not net.use_attention else net.attention_resolutions

    log('[load] create guided_unet...')
    model = create_model(net)
    model.change_input_conv()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['ema'], strict=False)
    log(f'[load] ckpt step={ck["step"]} missing={len(missing)} unexpected={len(unexpected)}')
    model = model.cuda().eval()

    to_tensor = transforms.ToTensor()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    inp = Path(args.input)
    imgs = sorted(inp.glob('*.jpg')) + sorted(inp.glob('*.png')) if inp.is_dir() else [inp]
    imgs = list(dict.fromkeys(imgs))
    log(f'[boot] {len(imgs)} images, up={args.up}')

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
        # 放大模式
        if args.up > 1:
            im = im.resize((W * args.up, H * args.up), Image.BICUBIC)
        W, H = im.size
        # 块网格 (含重叠)
        step = TILE - OVERLAP
        canvas = np.zeros((H, W, 3), dtype=np.float64)
        weight = np.zeros((H, W, 1), dtype=np.float64)
        n = 0
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                x1 = min(x0 + TILE, W)
                y1 = min(y0 + TILE, H)
                tile = im.crop((x0, y0, x1, y1))
                c_t = to_tensor(tile).unsqueeze(0).cuda() * 2 - 1  # [1,3,h,w] [-1,1]
                with torch.no_grad():
                    x0_t = np.sqrt(1 - SIGMA_P ** 2) * c_t + SIGMA_P * torch.randn_like(c_t)
                    t = torch.ones(1).cuda() * (T_ONE * 999)
                    inp_t = torch.cat([x0_t, c_t], dim=1)
                    v = model(inp_t, t)[:, :3]
                    sample = x0_t + (1.0 - EPS) * v
                sample = torch.clamp(sample, -1, 1)
                out_pil = transforms.ToPILImage()(sample[0].cpu() * 0.5 + 0.5)
                # 羽化叠加
                arr = np.asarray(out_pil).astype(np.float64)
                m = np.ones((y1 - y0, x1 - x0, 1))
                if x0 > 0:
                    m[:, :OVERLAP] = np.linspace(0, 1, OVERLAP)[None, :, None]
                if y0 > 0:
                    m[:OVERLAP, :] *= np.linspace(0, 1, OVERLAP)[:, None, None]
                canvas[y0:y1, x0:x1] += arr * m
                weight[y0:y1, x0:x1] += m
                n += 1
        canvas = canvas / np.maximum(weight, 1e-6)
        # 下采样回原尺寸
        if args.up > 1:
            pil = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
            pil = pil.resize((W // args.up, H // args.up), Image.LANCZOS)
            pil.save(outp)
        else:
            Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(outp)
        log(f'[done] {stem} ({n} tiles, {time.time()-t0:.0f}s)')
    log('[done] all')


if __name__ == '__main__':
    main()
