# -*- coding: utf-8 -*-
"""0.5B 版 LoRA scale 反推 (2026-09-12, 移植自 remerge_scale_14b.py)
W_merged(s) = W_base + (s/s0) * (W_merged(s0) - W_base)   —— 线性外推, 严格等价
用途: 从已加固的 M24 包(sc0.8) 生成 sc0.4 的 model_dir (与 sc0.4 出图同一模型主体)
用法: python remerge_scale_05b.py --src-pkg <pt> --scale 0.4 --out <pt> [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_SD = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os/checkpoints/ema_model.safetensors'


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-pkg', required=True)
    ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default=BASE_SD)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    log(f'读源包 {args.src_pkg}')
    pkg = torch.load(args.src_pkg, map_location='cpu', mmap=True, weights_only=False)
    s0 = float(pkg['lora_scale'])
    f = args.scale / s0
    log(f'源 s0={s0} -> 目标 s={args.scale} (系数 f={f:.6f})')
    log(f'源包键: {list(pkg.keys())}')

    dt = pkg['diT']
    out_diT, n_keys, maxdiff = {}, 0, 0.0
    with safe_open(args.base, framework='pt', device='cpu') as bf:
        bkeys = set(bf.keys())
        missing = [k for k in dt.keys() if k not in bkeys]
        if missing:
            log(f'⚠️ 基座缺 {len(missing)} 键 (前3: {missing[:3]}) -> 原样保留')
        for k, m in dt.items():
            if k not in bkeys:
                out_diT[k] = m.clone(); continue
            b = bf.get_tensor(k)
            if (not m.is_floating_point()) or (not b.is_floating_point()):
                out_diT[k] = m.clone(); continue
            m32, b32 = m.float(), b.float()
            new = (b32 + f * (m32 - b32)).to(m.dtype)
            out_diT[k] = new
            if args.check:
                maxdiff = max(maxdiff, float((new.float() - m32).abs().max()))
            n_keys += 1
            del b, m32, b32, new
    log(f'重建 {n_keys}/{len(dt)} 键')
    if args.check:
        log(f'[check] 恒等重建最大偏差 = {maxdiff:.3e} ({("恒等✅" if maxdiff == 0 else "非零")})')

    out = dict(pkg)                      # 保留 0.5B 的所有键 (format/diT_cfg/vae/dec/venc/...)
    out['diT'] = out_diT
    out['lora_scale'] = args.scale
    out['remerge_from'] = os.path.basename(args.src_pkg)
    out['remerge_src_scale'] = s0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    log(f'[save] {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB)')
    log('REMERGE_DONE')


if __name__ == '__main__':
    main()
