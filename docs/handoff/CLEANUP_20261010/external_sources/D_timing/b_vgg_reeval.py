# -*- coding: utf-8 -*-
"""VGG-lpips 重评 (2026-08-26): 用 net='vgg' 正确参数对已渲染的 val_vgg_out/* 重评
用法: D:/Python/python.exe D:/tmp_timing/b_vgg_reeval.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:/tmp_timing')
import pyiqa  # noqa: E402

VAL_DIR = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
OUT_ROOT = Path(r'D:/tmp_timing/val_vgg_out')
VARIANTS = ['v7_wavelet', 'v7_nofix', 'v7_adain', 'v6_wavelet', 'v5_wavelet']


def load_tensor(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).clamp(1e-4, 1 - 1e-4)


def lpips_tiled(metric, a_t, b_t, tile=1024, device='cuda'):
    """分块 LPIPS (4K 整图 OOM 时用); 返回全图均值"""
    vals = []
    _, _, H, W = a_t.shape
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            a2 = a_t[:, :, y:y + tile, x:x + tile]
            b2 = b_t[:, :, y:y + tile, x:x + tile]
            with torch.no_grad():
                vals.append(float(metric(a2, b2).mean().item()))
    return float(np.mean(vals))


def main():
    dev = 'cuda'
    clip_m = pyiqa.create_metric('clipiqa', device=dev)
    lp_alex = pyiqa.create_metric('lpips', device=dev)
    lp_vgg = pyiqa.create_metric('lpips', device=dev, net='vgg')
    niqe_m = pyiqa.create_metric('niqe', device=dev)

    results = {}
    for name in VARIANTS:
        src = OUT_ROOT / name
        if not src.is_dir():
            print(f'[skip] {name} missing', flush=True)
            continue
        per = {}
        for gtp in sorted(VAL_DIR.glob('*_gt.*')):
            cname = gtp.stem.replace('_gt', '')
            cands = [p for p in src.iterdir() if p.stem.startswith(cname) and '_lq' in p.stem]
            if not cands:
                continue
            ot = load_tensor(cands[0]).to(dev)
            gt = load_tensor(gtp).to(dev)
            with torch.no_grad():
                cq = float(clip_m(ot).mean().item())
                la = float(lp_alex(ot, gt).mean().item())
                lv = lpips_tiled(lp_vgg, ot, gt, device=dev)
                nq = float(niqe_m(ot).mean().item())
            per[cname] = {'clipiqa': round(cq, 4), 'lpips_alex': round(la, 4),
                          'lpips_vgg': round(lv, 4), 'niqe': round(nq, 3)}
            print(f'[{name}] {cname}: clip={cq:.4f} alex={la:.4f} vgg={lv:.4f} niqe={nq:.2f}', flush=True)
        results[name] = {'per': per, 'mean': {
            'clipiqa': round(float(np.mean([v['clipiqa'] for v in per.values()])), 4),
            'lpips_alex': round(float(np.mean([v['lpips_alex'] for v in per.values()])), 4),
            'lpips_vgg': round(float(np.mean([v['lpips_vgg'] for v in per.values()])), 4),
        }}
    OUT_ROOT.joinpath('val_vgg_final.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('[done] val_vgg_final.json', flush=True)


if __name__ == '__main__':
    main()
