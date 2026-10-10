# -*- coding: utf-8 -*-
"""LORA-SCREEN：在**同一编排**（交付 runner 的真实 512 路径）下，对 LoRA 变体/scale/seed 生成候选图。

用法: python lora_screen.py --out <dir> --lora <path> --scale S --seed N [--crops 512|full] [--cases ...]
说明：
  - 使用 eager 同配方（VAE fp16 + DiT/DINO fp16），与 OPT-01 部署一致；
  - 复用交付 runner 的完整 512 路径（分块/融合/wavelet/像素编码）；
  - 输出 JPEG（quality 98、subsampling 0）供官方风格 NR 指标评估；
  - 只做筛选，不产出交付件。
"""
import argparse
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
import delivery_runner as DR  # noqa: E402
import frozen_mode  # noqa: E402
import pixel_io  # noqa: E402
import r_verify  # noqa: E402

SRC = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集')
DEFAULT_CASES = [1, 10, 11, 12, 13, 100]
JPEG = {'quality': 98, 'subsampling': 0, 'optimize': False, 'progressive': False}


def load_case(case, mode):
    im = Image.open(SRC / ('case%d.jpg' % case)).convert('RGB')
    if mode == '512':
        w, h = im.size
        im = im.crop((w // 2 - 256, h // 2 - 256, w // 2 + 256, h // 2 + 256))
    arr = np.asarray(im)
    x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return (x * 2.0 - 1.0), im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--lora', required=True)
    ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--base', default=None)
    ap.add_argument('--crops', default='512', choices=['512', 'full'])
    ap.add_argument('--cases', default=','.join(str(c) for c in DEFAULT_CASES))
    a = ap.parse_args()
    frozen_mode.apply_frozen_mode()
    dev = torch.device('cuda')
    cfg = PBcore.load_config()
    cfg['lora_path'] = a.lora
    cfg['lora_scale'] = float(a.scale)
    if a.base:
        cfg['base_weights'] = a.base
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {'schema': 'LORA-screen/1', 'lora': a.lora, 'scale': a.scale, 'seed': a.seed,
           'base': cfg['base_weights'], 'crops': a.crops, 'images': [], 'started_at':
           time.strftime('%Y-%m-%d %H:%M:%S')}
    t0 = time.perf_counter()
    bundle, patched = r_verify.build_eager(cfg, dev, vae_dtype=torch.float16,
                                           work_dtype=torch.float16)
    rec['build_s'] = round(time.perf_counter() - t0, 1)
    # 用交付 runner 的真实编排（模型替换为 eager bundle，seed 覆盖）
    runner = DR.Runner(str(Path(r'D:\models\opt01\model_dir')), device='cuda')
    runner.model = bundle
    runner.seed = int(a.seed)
    for case in [int(s) for s in a.cases.split(',') if s.strip()]:
        x, wh = load_case(case, a.crops)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            y = runner.infer(x, prompt='')
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000.0
        u8 = pixel_io.to_u8(y.detach().cpu())
        p = out / ('case%d.jpg' % case)
        Image.fromarray(u8).save(p, format='JPEG', **JPEG)
        rec['images'].append({'case': case, 'wall_ms': round(ms, 1), 'wh': wh,
                              'out_sha256': __import__('hashlib').sha256(
                                  np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest()})
        print('  case%-4d %6.0fms %s' % (case, ms, wh), flush=True)
    rec['total_s'] = round(time.perf_counter() - t0, 1)
    (out / 'screen_meta.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                                          encoding='utf-8')
    print('DONE', out, rec['total_s'], 's')


if __name__ == '__main__':
    main()
