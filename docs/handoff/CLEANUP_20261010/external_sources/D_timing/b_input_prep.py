# -*- coding: utf-8 -*-
"""输入预处理判别实验 (2026-08-31): VOSR 的输入分布改变能否压低 lpips?
A: 输入 = s6k CNN 预去噪输出 (深谷噪声图的幻觉压力应降低)
B: 输入 = VOSR 自身输出 (自精炼, 二遍)
对照: M24 (输入 = LQ), val 0.9193/0.3789
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
OUT_ROOT = Path(r'D:/tmp_timing/val_input_out')
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
PY = r'D:/Python/python.exe'
INFER = r'D:/tmp_timing/vosr_lora_infer.py'
S6K_VAL = r'D:/tmp_timing/cnn_val_s6000'
M24_VAL = r'D:/tmp_timing/val2x_out/u1'

ENV = dict(os.environ)
ENV['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
ENV['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

VARIANTS = [
    ('s6k_in', S6K_VAL),   # 输入 = s6k 预去噪
    ('selfref_in', M24_VAL),  # 输入 = VOSR 自身输出
]


def render(name, in_dir):
    out = OUT_ROOT / name
    if out.exists() and any(out.iterdir()):
        print(f'[skip] {name}', flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, INFER, '-c', CKPT, '--lora', LORA, '--lora-scale', '0.8',
           '-i', in_dir, '-o', str(out), '-u', '1', '--seed', '8',
           '--align_method', 'wavelet', '--tile_size', '512',
           '--vae_tile_size', '1024', '--vae_tile_overlap', '64']
    t0 = time.time()
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'[FAIL] {name}', flush=True)
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
    lp_m = pyiqa.create_metric('lpips', device=dev)
    psnr_m = pyiqa.create_metric('psnr', device=dev)
    ssim_m = pyiqa.create_metric('ssim', device=dev)

    results = {}
    for name, in_dir in VARIANTS:
        out = render(name, in_dir)
        per = {}
        # 输入文件命名映射: s6k -> caseN_pred.jpg; selfref -> caseN_lq.png
        for gtp in sorted(VAL_DIR.glob('*_gt.*')):
            cname = gtp.stem.replace('_gt', '')
            cands = [p for p in out.iterdir() if p.stem.startswith(cname) and p.suffix == '.png']
            if not cands:
                print(f'[warn] {name}: {cname} no output', flush=True)
                continue
            ot = load_tensor(cands[0]).to(dev)
            gt = load_tensor(gtp).to(dev)
            with torch.no_grad():
                cq = float(clip_m(ot).mean().item())
                lp = float(lp_m(ot, gt).mean().item())
                ps = float(psnr_m(ot, gt).mean().item())
                ss = float(ssim_m(ot, gt).mean().item())
            per[cname] = {'clipiqa': round(cq, 4), 'lpips': round(lp, 4),
                          'psnr': round(ps, 3), 'ssim': round(ss, 4)}
            print(f'[{name}] {cname}: clip={cq:.4f} lpips={lp:.4f} psnr={ps:.2f} ssim={ss:.4f}', flush=True)
        results[name] = {'per': per, 'mean': {
            'clipiqa': round(float(np.mean([v['clipiqa'] for v in per.values()])), 4),
            'lpips': round(float(np.mean([v['lpips'] for v in per.values()])), 4),
            'psnr': round(float(np.mean([v['psnr'] for v in per.values()])), 3),
            'ssim': round(float(np.mean([v['ssim'] for v in per.values()])), 4),
        }}
        print(f'[{name}] MEAN: {results[name]["mean"]}', flush=True)
    OUT_ROOT.joinpath('val_input.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('[done] val_input.json', flush=True)


if __name__ == '__main__':
    main()
