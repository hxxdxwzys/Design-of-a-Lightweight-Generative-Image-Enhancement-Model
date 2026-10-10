# -*- coding: utf-8 -*-
"""BASE-SCREEN：对「基座/步数」等能力层候选做 512 快速代理筛选（不改交付件）。

用法: python base_screen.py --out DIR --base CKPT [--lora PATH --scale S] [--steps N] [--cases ...] [--seed 42]
说明：单块 512 裁切路径（与交付 runner 同语义：encode → features → N 步 → decode → wavelet）。
"""
import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline_baseline'))
import core as PBcore  # noqa: E402
import frozen_mode  # noqa: E402
import pixel_io  # noqa: E402
import r_verify  # noqa: E402

SRC = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集')
JPEG = {'quality': 98, 'subsampling': 0, 'optimize': False, 'progressive': False}


def load_case(case, crops='512'):
    im = Image.open(SRC / ('case%d.jpg' % case)).convert('RGB')
    if crops == '512':
        w, h = im.size
        im = im.crop((w // 2 - 256, h // 2 - 256, w // 2 + 256, h // 2 + 256))
    arr = np.asarray(im)
    x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x * 2.0 - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--lora', default=None)
    ap.add_argument('--scale', type=float, default=0.0)
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--cases', default='1,10,11,12,13,100')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    frozen_mode.apply_frozen_mode()
    dev = torch.device('cuda')
    cfg = PBcore.load_config()
    cfg['base_weights'] = a.base
    if a.lora:
        cfg['lora_path'] = a.lora
        cfg['lora_scale'] = float(a.scale)
    else:
        cfg['lora_scale'] = 0.0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {'schema': 'BASE-screen/1', 'base': a.base, 'lora': a.lora, 'scale': a.scale,
           'steps': a.steps, 'seed': a.seed, 'images': []}
    t0 = time.perf_counter()
    try:
        bundle, _ = r_verify.build_eager(cfg, dev, vae_dtype=torch.float16,
                                         work_dtype=torch.float16)
    except Exception as e:
        rec['error'] = '%s: %s' % (type(e).__name__, e)
        (out / 'screen_meta.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                                              encoding='utf-8')
        print('BUILD_FAIL', rec['error'], flush=True)
        return 1
    rec['build_s'] = round(time.perf_counter() - t0, 1)
    n = max(int(a.steps), 1)
    sched = [(1.0 - i / n, 1.0 - (i + 1) / n) for i in range(n)]
    for case in [int(s) for s in a.cases.split(',') if s.strip()]:
        x = load_case(case).to(dev)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            xh = x.to(torch.float16)          # 与交付 runner 一致：模块输入按 weight_dtype 转换
            lq = bundle.enc(xh)
            feats = bundle.features(xh)
            g = torch.Generator(device=dev)
            g.manual_seed(int(a.seed))
            z = torch.randn(lq.shape, generator=g, device=dev, dtype=torch.float32).to(lq.dtype)
            for tc, tn in sched:
                t_cur = torch.full((1,), float(tc), device=dev, dtype=lq.dtype)
                t_nxt = torch.full((1,), float(tn), device=dev, dtype=lq.dtype)
                z = bundle.dit(lq, z, t_cur, t_nxt, feats)
            sr = bundle.dec(z).clamp(-1.0, 1.0)
            y = bundle.wav(sr, x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000.0
        u8 = pixel_io.to_u8(y.detach().cpu())
        p = out / ('case%d.jpg' % case)
        Image.fromarray(u8).save(p, format='JPEG', **JPEG)
        fin = bool(torch.isfinite(y).all())
        rec['images'].append({'case': case, 'wall_ms': round(ms, 1), 'finite': fin})
        print('  case%-4d %6.0fms finite=%s' % (case, ms, fin), flush=True)
    rec['total_s'] = round(time.perf_counter() - t0, 1)
    (out / 'screen_meta.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                                          encoding='utf-8')
    print('DONE', out, rec['total_s'], 's', 'build', rec['build_s'], 's')


if __name__ == '__main__':
    main()
