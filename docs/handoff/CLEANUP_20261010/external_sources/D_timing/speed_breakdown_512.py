# -*- coding: utf-8 -*-
"""512 前向时延分解 + 剪枝收益实测 (2026-09-12)
================================================
背景: 阶段 3 要把加速比抬到 >1 (D3 零覆盖区), 但我早先口头给的"剪枝 28->7 层 ≈ 4-8x"
      没有依据 (隐含假设 DiT 占满全部时延)。本脚本把那个猜测变成测量:
  ① 全链 512 时延 (对标旧基线 285.6ms)
  ② 四组件分解: DINOv2 / VAE encode / DiT / LightDecoder
  ③ DiT 深度截断 28->21/14/7 的实测时延 -> 直接给出剪枝收益曲线
必须在【安静 GPU】上跑 (有并发渲染时时延无意义)。

用法: python speed_breakdown_512.py [--lora <lora.pt>] [--scale 0.8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
os.environ.setdefault('VOSR_CACHE', r'C:/models/finals/preset/ckpts/torch_cache')
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

VOSR_DIR = r'D:/models/finals/VOSR'
sys.path.insert(0, VOSR_DIR)
sys.path.insert(0, r'D:/tmp_timing')

import torch  # noqa: E402
import numpy as np  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from inference_vosr_onestep import (  # noqa: E402
    load_config_with_cli, load_dinov2, get_venc_features,
)

CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
SD2_AE = r'C:/models/finals/preset/ckpts/stable-diffusion-2-1-base'
LWDEC = r'C:/models/finals/preset/ckpts/sd21_lwdecoder.pth'
OUT = Path(r'D:/tmp_timing/speed_breakdown_512.json')
BASELINE_MS = 285.6      # 旧世界同机实测 (1.40ms x 204x); OSEDiff 权重已丢失, 不可复测


def timeit(fn, n=10, warm=3, label=''):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    med = ts[len(ts) // 2]
    print(f'  {label:<34} 中位 {med:8.1f} ms   最快 {ts[0]:8.1f} ms', flush=True)
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora', default=r'D:/models/vosr_lora/v7/lora_s5000.pt')
    ap.add_argument('--scale', type=float, default=0.8)
    ap.add_argument('--n', type=int, default=10)
    a = ap.parse_args()

    from models.lightningdit import LightningDiT
    from models.light_decoder import LightDecoder
    from diffusers import AutoencoderKL
    from peft import LoraConfig, get_peft_model

    torch.hub.set_dir(r'C:/models/finals/preset/ckpts/torch_cache')
    sys.argv = [sys.argv[0]]
    import argparse as _ap
    p2 = _ap.ArgumentParser()
    p2.add_argument('--checkpoint', '-c', default=CKPT)
    p2.add_argument('--config', default='')
    args = load_config_with_cli(CKPT, p2)
    if getattr(args, 'auxiliary_time_cond', None) is None:
        args.auxiliary_time_cond = True
    args.auxiliary_time_cond = bool(args.auxiliary_time_cond)

    model = LightningDiT(
        input_size=args.resolution // 8, patch_size=args.patch_size,
        in_channels=2 * 4, out_channels=4, hidden_size=args.dim, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, z_dims=args.enc_dim,
        encdim_ratio=args.encdim_ratio, auxiliary_time_cond=args.auxiliary_time_cond,
        use_qknorm=args.use_qknorm, use_swiglu=args.use_swiglu, use_rope=args.use_rope,
        use_rmsnorm=args.use_rmsnorm, wo_shift=args.wo_shift,
        num_fused_layers=len(args.layer_dinov2b_list))
    sd = load_file(os.path.join(CKPT, 'checkpoints', 'ema_model.safetensors'))
    model.load_state_dict(sd, strict=False)
    model.forward = model.forward_flexible
    if a.lora and a.scale != 1.0:
        lora = torch.load(a.lora, map_location='cpu', weights_only=False)
        cfg = json.loads(lora['lora_config'])
        model = get_peft_model(model, LoraConfig(**cfg))
        lsd = {k: (v * a.scale if 'lora_B' in k else v)
               for k, v in lora['state_dict'].items() if 'lora' in k}
        model.load_state_dict(lsd, strict=False)
        print(f'[lora] {Path(a.lora).name} scale={a.scale}', flush=True)
    model = model.half().cuda().eval()
    n_blocks = len(model.blocks) if hasattr(model, 'blocks') else len(model.base_model.model.blocks)
    print(f'[model] dim={args.dim} depth={args.depth} blocks={n_blocks} '
          f'enc={args.enc_type} enc_dim={args.enc_dim} res={args.resolution}', flush=True)

    vae = AutoencoderKL.from_pretrained(SD2_AE, subfolder='vae').cuda().eval()
    lw = torch.load(LWDEC, map_location='cpu', weights_only=False)
    dec = LightDecoder(**{k: lw['config'][k] for k in
                          ('in_channels', 'out_channels', 'block_out_channels', 'layers_per_block')})
    dec.load_state_dict(lw['model_state_dict'])
    dec = dec.half().cuda().eval()
    venc = load_dinov2(args, 'cuda').half()

    x = (torch.rand(1, 3, 512, 512).cuda() * 2 - 1)
    z = torch.randn(1, 4, 64, 64).cuda().half()
    t = torch.ones(1, device='cuda')
    t0v = torch.zeros(1, device='cuda')

    def full():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            zf = [q.detach() for q in get_venc_features(venc, x, args)]
            lat = vae.encode(x.half()).latent_dist.sample() * vae.config.scaling_factor
            u = model(torch.cat([lat, z], 1), t, t0v, zf)
            return dec((z - u) / vae.config.scaling_factor).clamp(-1, 1)

    print('\n=== ① 全链 ===', flush=True)
    full_ms = timeit(full, a.n, 3, '全链 (DINOv2+VAE+DiT+dec)')

    print('\n=== ② 组件分解 ===', flush=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        zf = [q.detach() for q in get_venc_features(venc, x, args)]
        lat = vae.encode(x.half()).latent_dist.sample() * vae.config.scaling_factor
        u = model(torch.cat([lat, z], 1), t, t0v, zf)
        z1 = (z - u) / vae.config.scaling_factor

    dino_ms = None
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        dino_ms = timeit(lambda: get_venc_features(venc, x, args), a.n, 3, 'DINOv2 特征 (448)')
        enc_ms = timeit(lambda: vae.encode(x.half()).latent_dist.sample(), a.n, 3, 'VAE encode (512->64)')
        dit_ms = timeit(lambda: model(torch.cat([lat, z], 1), t, t0v, zf), a.n, 3,
                        f'DiT 1 步 ({n_blocks} blocks)')
        dec_ms = timeit(lambda: dec(z1), a.n, 3, 'LightDecoder decode')

    print('\n=== ③ DiT 深度截断 (剪枝收益实测) ===', flush=True)
    m = model.base_model.model if hasattr(model, 'base_model') else model
    all_blocks = list(m.blocks)
    prune = {}
    for keep in (21, 14, 7, 4):
        if keep >= len(all_blocks):
            continue
        m.blocks = torch.nn.ModuleList(all_blocks[:keep])
        ms = None
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            ms = timeit(lambda: model(torch.cat([lat, z], 1), t, t0v, zf), a.n, 2,
                        f'DiT {keep} blocks')
        prune[keep] = ms
    m.blocks = torch.nn.ModuleList(all_blocks)

    # DiT 按比例缩放后的全链估计 (其余组件不变)
    est = {}
    for keep, ms in prune.items():
        tot = full_ms - dit_ms + ms
        est[keep] = {'dit_ms': round(ms, 1), 'full_est_ms': round(tot, 1),
                     'ratio_vs_baseline': round(BASELINE_MS / tot, 2)}

    print('\n=== ④ 结论 ===', flush=True)
    print(f'  基线(历史) {BASELINE_MS} ms | 我们全链 {full_ms:.1f} ms -> 当前比 {BASELINE_MS/full_ms:.2f}x')
    parts = {'DINOv2': dino_ms, 'VAE_encode': enc_ms, 'DiT': dit_ms, 'LightDecoder': dec_ms}
    s = sum(parts.values())
    for k, v in parts.items():
        print(f'    {k:<14}{v:8.1f} ms  占 {v/s*100:5.1f}%')
    print(f'    组件和 {s:.1f} vs 全链 {full_ms:.1f} (差 {full_ms-s:+.1f} = 调度/拷贝开销)')
    print('\n  剪枝后全链估计:')
    for keep, d in est.items():
        print(f'    keep {keep:2d}/{n_blocks} blocks -> 全链 ~{d["full_est_ms"]:.0f} ms '
              f'-> 比基线 {d["ratio_vs_baseline"]:.2f}x')

    OUT.write_text(json.dumps({
        'baseline_ms': BASELINE_MS, 'full_ms': round(full_ms, 1),
        'ratio_now': round(BASELINE_MS / full_ms, 3),
        'parts_ms': {k: round(v, 1) for k, v in parts.items()},
        'dit_prune': {str(k): v for k, v in est.items()},
        'model': {'dim': args.dim, 'depth': args.depth, 'blocks': n_blocks,
                  'enc': args.enc_type, 'enc_dim': args.enc_dim},
        'lora': a.lora, 'scale': a.scale,
    }, indent=1), encoding='utf-8')
    print(f'\n已存 {OUT}')
    print('SPEED_BREAKDOWN_DONE')


if __name__ == '__main__':
    main()
