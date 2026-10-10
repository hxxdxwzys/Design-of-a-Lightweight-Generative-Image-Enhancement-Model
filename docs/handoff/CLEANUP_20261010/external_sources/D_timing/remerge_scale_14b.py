# -*- coding: utf-8 -*-
"""从已合并的合规 pkg 反推任意 LoRA scale 档 (2026-09-10)

动机: 波 2 需要的 m0s1-sc0.90 / sc0.97 只有 P100 服务器上有 LoRA 原件, 而服务器不可达。
但已提交的 Track2_14B_M0S1_SC093.zip 里的 your_model.pt 是**合并后**权重, 而基座
ema_model.safetensors 在本地 -> LoRA 增量 Δ = merged - base 可逐元素恢复。

数学 (merge_14b.py 的约定: lora_sd['lora_B'] *= scale 后再 merge):
    W_merged(s0) = W_base + s0 * (alpha/r) * B @ A
    => Δ := W_merged(s0) - W_base = s0 * (alpha/r) * B @ A
    => W_base + (s/s0) * Δ = W_base + s * (alpha/r) * B @ A   == 真实 LoRA @scale s
即 **线性外推严格等价**, 与 LoRA 的分解形式无关 (只需加性)。

已知代价: pkg 里 diT 是 fp16 存的, 基座是 fp32 -> Δ 里含 fp16 量化噪声。
实测 (2026-09-10, m0s1-sc0.93): ‖Δ‖rms/‖base‖rms = 0.68%, 重建 SNR = 27.7 dB
(大能量键 31-32 dB)。是否影响输出 -> 由 val 3 张端到端验证 (预登记 (0.9209, 0.3556))。

用法:
  D:/Python/python.exe remerge_scale_14b.py --src-pkg <your_model.pt> --scale 0.90 --out <your_model.pt>
  D:/Python/python.exe remerge_scale_14b.py --src-pkg ... --scale 0.93 --out ... --check
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from safetensors import safe_open

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_SD = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os/checkpoints/ema_model.safetensors'


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-pkg', required=True, help='源 your_model.pt (含 diT + lora_scale)')
    ap.add_argument('--scale', type=float, required=True, help='目标 LoRA scale')
    ap.add_argument('--out', required=True, help='输出 your_model.pt')
    ap.add_argument('--base', default=BASE_SD)
    ap.add_argument('--check', action='store_true',
                    help='目标 scale == 源 scale 时逐元素校验重建是否为恒等')
    args = ap.parse_args()

    log(f'读源包 {args.src_pkg}')
    pkg = torch.load(args.src_pkg, map_location='cpu', mmap=True, weights_only=False)
    s0 = float(pkg['lora_scale'])
    f = args.scale / s0
    log(f'源 scale s0={s0} -> 目标 s={args.scale}  (外推系数 f={f:.6f})')

    dt = pkg['diT']
    out_diT = {}
    maxdiff = 0.0
    n_keys = 0
    with safe_open(args.base, framework='pt', device='cpu') as bf:
        bkeys = set(bf.keys())
        missing = [k for k in dt.keys() if k not in bkeys]
        if missing:
            log(f'⚠️ 基座缺 {len(missing)} 个键 (前3: {missing[:3]}) -> 这些键原样保留')
        for k, m in dt.items():
            if k not in bkeys:
                out_diT[k] = m.clone()
                continue
            b = bf.get_tensor(k)
            if (not m.is_floating_point()) or (not b.is_floating_point()):
                out_diT[k] = m.clone()
                continue
            m32 = m.float()
            d = m32 - b.float()
            new = (b.float() + f * d).to(m.dtype)
            out_diT[k] = new
            if args.check:
                maxdiff = max(maxdiff, float((new.float() - m32).abs().max()))
            n_keys += 1
            del b, m32, d, new

    log(f'重建 {n_keys}/{len(dt)} 键, 输出 dtype={next(iter(out_diT.values())).dtype}')
    if args.check:
        log(f'[check] 恒等重建最大逐元素偏差 = {maxdiff:.3e} '
            f'({"✅ 恒等" if maxdiff == 0 else "⚠️ 非零"})')

    out = {
        'format': pkg['format'],
        'diT_cfg': pkg['diT_cfg'],
        'diT': out_diT,
        'vae': {k: v for k, v in pkg['vae'].items()},
        'vae_cfg': pkg['vae_cfg'],
        'venc': {k: v for k, v in pkg['venc'].items()},
        'dinov2_size': pkg['dinov2_size'],
        'layer_dinov2b_list': pkg['layer_dinov2b_list'],
        'lora_scale': args.scale,
        'z_dim': pkg['z_dim'],
        'remerge_from': os.path.basename(args.src_pkg),
        'remerge_src_scale': s0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    log(f'[save] {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB)')
    log('REMERGE_DONE')


if __name__ == '__main__':
    main()
