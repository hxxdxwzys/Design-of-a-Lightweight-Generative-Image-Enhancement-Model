# -*- coding: utf-8 -*-
"""VGG-lpips + 对齐方法判别实验 (2026-08-26)
问题1: 官方 lpips 若用 VGG 骨干, 池内源在 VGG 下的相对排序是否不同于 alex?
问题2: v7 在不同 align (wavelet/nofix/adain) 下 lpips/clip 差异?
渲染: v7_seed8 {wavelet,nofix,adain} + v6_s4000 wavelet + v5_s4000_sc0.8 wavelet (val 3 张)
评测: clipiqa + lpips(alex) + lpips(vgg) + niqe
"""
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
OUT_ROOT = Path(r'D:/tmp_timing/val_vgg_out')
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
PY = r'D:/Python/python.exe'
INFER = r'D:/tmp_timing/vosr_lora_infer.py'

ENV = dict(os.environ)
ENV['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
ENV['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

VARIANTS = [
    ('v7_wavelet', dict(lora=LORA, scale='0.8', seed='8', align='wavelet')),
    ('v7_nofix', dict(lora=LORA, scale='0.8', seed='8', align='nofix')),
    ('v7_adain', dict(lora=LORA, scale='0.8', seed='8', align='adain')),
    ('v6_wavelet', dict(lora=r'D:/models/vosr_lora/v6/lora_s4000.pt', scale='1.0', seed='42', align='wavelet')),
    ('v5_wavelet', dict(lora=r'D:/models/vosr_lora/v5/lora_s4000.pt', scale='0.8', seed='42', align='wavelet')),
]


def render(name, cfg):
    out = OUT_ROOT / name
    if out.exists() and any(out.iterdir()):
        print(f'[skip] {name}', flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, INFER, '-c', CKPT, '--lora', cfg['lora'], '--lora-scale', cfg['scale'],
           '-i', str(LQ_DIR), '-o', str(out), '-u', '1', '--seed', cfg['seed'],
           '--align_method', cfg['align'], '--tile_size', '512',
           '--vae_tile_size', '1024', '--vae_tile_overlap', '64']
    t0 = time.time()
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'[FAIL] {name}: rc={r.returncode}', flush=True)
        print(r.stderr[-3000:], flush=True)
        sys.exit(1)
    print(f'[render] {name} {(time.time()-t0)/60:.1f}min', flush=True)
    return out


def load_tensor(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).clamp(1e-4, 1 - 1e-4)


def main():
    sys.path.insert(0, r'D:/tmp_timing')
    import pyiqa
    dev = 'cuda'
    clip_m = pyiqa.create_metric('clipiqa', device=dev)
    lp_alex = pyiqa.create_metric('lpips', device=dev)          # default alex
    lp_vgg = pyiqa.create_metric('lpips', device=dev, pretrained_model='vgg')  # vgg backbone
    niqe_m = pyiqa.create_metric('niqe', device=dev)

    results = {}
    for name, cfg in VARIANTS:
        out = render(name, cfg)
        per = {}
        for gtp in sorted(VAL_DIR.glob('*_gt.*')):
            cname = gtp.stem.replace('_gt', '')
            cands = [p for p in out.iterdir() if p.stem.startswith(cname) and '_lq' in p.stem]
            if not cands:
                continue
            ot = load_tensor(cands[0]).to(dev)
            gt = load_tensor(gtp).to(dev)
            with torch.no_grad():
                cq = float(clip_m(ot).mean().item())
                la = float(lp_alex(ot, gt).mean().item())
                lv = float(lp_vgg(ot, gt).mean().item())
                nq = float(niqe_m(ot).mean().item())
            per[cname] = {'clipiqa': round(cq, 4), 'lpips_alex': round(la, 4),
                          'lpips_vgg': round(lv, 4), 'niqe': round(nq, 3)}
            print(f'[{name}] {cname}: clip={cq:.4f} alex={la:.4f} vgg={lv:.4f} niqe={nq:.2f}', flush=True)
        results[name] = {'per': per, 'mean': {
            'clipiqa': round(float(np.mean([v['clipiqa'] for v in per.values()])), 4),
            'lpips_alex': round(float(np.mean([v['lpips_alex'] for v in per.values()])), 4),
            'lpips_vgg': round(float(np.mean([v['lpips_vgg'] for v in per.values()])), 4),
        }}
    OUT_ROOT.joinpath('val_vgg.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('[done] val_vgg.json', flush=True)


if __name__ == '__main__':
    main()
