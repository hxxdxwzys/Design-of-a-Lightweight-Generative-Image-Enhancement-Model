# -*- coding: utf-8 -*-
"""2x val discriminative experiment (2026-08-26): M24 config (v7 lora_s5000 sc0.8 seed8 wavelet)
A/B: 1x render vs 2x render + downsample (BICUBIC / LANCZOS), per-case clipiqa/lpips/psnr/ssim.
Question: does native-2x generation (in-manifold) beat 1x on lpips where clip has headroom?
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

VAL_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
LQ_DIR = Path(r'D:/tmp_timing/val_lq_only')
OUT_ROOT = Path(r'D:/tmp_timing/val2x_out')
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
PY = r'D:/Python/python.exe'
INFER = r'D:/tmp_timing/vosr_lora_infer.py'

ENV = dict(os.environ)
ENV['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
ENV['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def render(upscale, out_dir, seed=8):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f'[skip] {out_dir} exists', flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, INFER, '-c', CKPT, '--lora', LORA, '--lora-scale', '0.8',
           '-i', str(LQ_DIR), '-o', str(out_dir), '-u', str(upscale),
           '--seed', str(seed), '--align_method', 'wavelet',
           '--tile_size', '512', '--vae_tile_size', '1024', '--vae_tile_overlap', '64']
    print(f'[render] u={upscale} -> {out_dir}', flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    if r.returncode != 0:
        print('[render FAILED] rc=', r.returncode, flush=True)
        print(r.stdout[-3000:], flush=True)
        print(r.stderr[-3000:], flush=True)
        sys.exit(1)
    print(f'[render done] {(time.time()-t0)/60:.1f}min', flush=True)


def downsample(src_dir, dst_dir, filt):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.glob('*.png')):
        with Image.open(p) as im:
            w, h = im.size
            im2 = im.resize((w // 2, h // 2), filt)
            im2.save(dst_dir / p.name)


def load_tensor(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).clamp(1e-4, 1 - 1e-4)


def evaluate(src_dir, label, out_json, device):
    sys.path.insert(0, r'D:/tmp_timing')
    import pyiqa
    clipiqa_m = pyiqa.create_metric('clipiqa', device=device)
    lpips_m = pyiqa.create_metric('lpips', device=device)
    psnr_m = pyiqa.create_metric('psnr', device=device)
    ssim_m = pyiqa.create_metric('ssim', device=device)

    src = Path(src_dir)
    per = {}
    for gtp in sorted(VAL_DIR.glob('*_gt.*')):
        name = gtp.stem.replace('_gt', '')
        cands = [p for p in src.iterdir() if p.stem.startswith(name) and '_lq' in p.stem]
        if not cands:
            print(f'[warn] {name}: no output', flush=True)
            continue
        out_t = load_tensor(cands[0]).to(device)
        gt_t = load_tensor(gtp).to(device)
        with torch.no_grad():
            cq = float(clipiqa_m(out_t).mean().item())
            lp = float(lpips_m(out_t, gt_t).mean().item())
            ps = float(psnr_m(out_t, gt_t).mean().item())
            ss = float(ssim_m(out_t, gt_t).mean().item())
        per[name] = {'clipiqa': round(cq, 4), 'lpips': round(lp, 4),
                     'psnr': round(ps, 3), 'ssim': round(ss, 4)}
        print(f'[{label}] {name}: clip={cq:.4f} lpips={lp:.4f} psnr={ps:.2f} ssim={ss:.4f}', flush=True)
    res = {
        'label': label,
        'mean': {
            'clipiqa': round(float(np.mean([v['clipiqa'] for v in per.values()])), 4),
            'lpips': round(float(np.mean([v['lpips'] for v in per.values()])), 4),
            'psnr': round(float(np.mean([v['psnr'] for v in per.values()])), 3),
            'ssim': round(float(np.mean([v['ssim'] for v in per.values()])), 4),
        },
        'per': per,
    }
    print(f'[{label}] MEAN: {res["mean"]}', flush=True)
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-render', action='store_true')
    args = ap.parse_args()

    device = 'cuda'
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    u1 = OUT_ROOT / 'u1'
    u2 = OUT_ROOT / 'u2'
    u2b = OUT_ROOT / 'u2_bicubic'
    u2l = OUT_ROOT / 'u2_lanczos'

    if not args.skip_render:
        render(1, u1)
        render(2, u2)
        print('[downsample] BICUBIC', flush=True)
        downsample(u2, u2b, Image.BICUBIC)
        print('[downsample] LANCZOS', flush=True)
        downsample(u2, u2l, Image.LANCZOS)

    res = {}
    res['u1'] = evaluate(u1, 'u1_1x', OUT_ROOT / 'b_2x_u1.json', device)
    res['u2_bicubic'] = evaluate(u2b, 'u2_bicubic', OUT_ROOT / 'b_2x_u2b.json', device)
    res['u2_lanczos'] = evaluate(u2l, 'u2_lanczos', OUT_ROOT / 'b_2x_u2l.json', device)
    (OUT_ROOT / 'b_2x_val.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('[done] b_2x_val.json', flush=True)


if __name__ == '__main__':
    main()
