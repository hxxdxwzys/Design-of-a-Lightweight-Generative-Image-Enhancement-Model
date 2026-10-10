# -*- coding: utf-8 -*-
"""1.4B fp16 vs fp32 渲染对齐验证 (2026-08-31, Phase 0 门控)

目标: 验证 vosr_lora_infer_14b.py 的 --fp16 渲染与 fp32 对齐 (val 1 张, diff<0.01),
    通过后 1.4B val 门控可从 22min/组合 (fp32) 降到 ~10min (fp16)。

用法 (cwd 必须 C:/models/finals, inference_vosr_onestep 相对路径依赖):
  cd C:/models/finals && D:/Python/python.exe D:/tmp_timing/b_14b_fp16_align.py \
      -i C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集/case1_lq.jpg \
      --lora D:/models/vosr_lora/v14/lora_s1000.pt --out D:/tmp_timing/fp16_align_out

输出: {out}/fp32.npy + {out}/fp16.npy + 终端对齐报告 (MAE/max/diff@0.01 判定)
"""
import argparse
import os
import sys
import time
from contextlib import nullcontext

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

VOSR_DIR = r'D:/models/finals/VOSR'
sys.path.insert(0, VOSR_DIR)
sys.path.insert(0, r'D:/tmp_timing')

import vosr_lora_infer_14b as inf  # noqa: E402  (复用 build_model + 常量)
from inference_vosr_onestep import (  # noqa: E402
    load_config_with_cli, load_dinov2, tiled_latent_inference,
)


def render_raw(args, lora_path, img_path, fp16, device='cuda'):
    """返回原始模型输出张量 [-1,1] float32 [1,3,H,W], 与 vosr_lora_infer_14b.py 同路径。"""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    model = inf.build_model(args, lora_path).to(device).eval()
    if fp16:
        model = model.half()  # 与生产脚本一致: 仅 model 显式 half, VAE/venc 保持 fp32 走 autocast

    if args.ae_type == 'qwen':
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        vae = AutoencoderKLQwenImage2D.from_pretrained(inf.QWEN_AE).to(device).eval()
    else:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(inf.SD2_AE, subfolder='vae').to(device).eval()
    venc = load_dinov2(args, device)

    raw_img = Image.open(img_path).convert('RGB')
    lq = transforms.ToTensor()(raw_img).unsqueeze(0).to(device) * 2.0 - 1.0
    with torch.no_grad():
        ctx = torch.autocast('cuda', dtype=torch.float16) if fp16 else nullcontext()
        with ctx:
            sr = tiled_latent_inference(model, vae, venc, lq, args, device=device)
    return sr[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_image', '-i', type=str, required=True)
    ap.add_argument('--lora', type=str, required=True)
    ap.add_argument('--output_dir', '-o', type=str, required=True)
    ap.add_argument('--checkpoint', '-c', type=str, default=r'C:/models/finals/preset/ckpts/VOSR_1.4B_os')
    ap.add_argument('--upscale', '-u', type=int, default=1)
    ap.add_argument('--tile_size', type=int, default=512)
    ap.add_argument('--tile_overlap', type=int, default=32)
    ap.add_argument('--vae_tile_size', type=int, default=1024)
    ap.add_argument('--vae_tile_overlap', type=int, default=None)
    ap.add_argument('--infer_steps', type=int, default=1)
    ap.add_argument('--config', type=str, default='')
    temp_args, _ = ap.parse_known_args()
    args = load_config_with_cli(temp_args.checkpoint, ap)
    if getattr(args, 'auxiliary_time_cond', None) is None:
        args.auxiliary_time_cond = True
    args.auxiliary_time_cond = bool(args.auxiliary_time_cond)

    os.makedirs(temp_args.output_dir, exist_ok=True)
    lora_path = torch.load(temp_args.lora, map_location='cpu', weights_only=False)
    print(f'[lora] {temp_args.lora}: variant={lora_path.get("variant")} step={lora_path.get("step")}', flush=True)

    t0 = time.time()
    print('[run] fp32 ...', flush=True)
    out32 = render_raw(args, lora_path, temp_args.input_image, fp16=False)
    np.save(os.path.join(temp_args.output_dir, 'fp32.npy'), out32)
    print(f'[run] fp32 done {time.time() - t0:.0f}s, saved fp32.npy', flush=True)

    t1 = time.time()
    print('[run] fp16 ...', flush=True)
    out16 = render_raw(args, lora_path, temp_args.input_image, fp16=True)
    np.save(os.path.join(temp_args.output_dir, 'fp16.npy'), out16)
    print(f'[run] fp16 done {time.time() - t1:.0f}s, saved fp16.npy', flush=True)

    a, b = out32.astype(np.float64), out16.astype(np.float64)
    mae = np.abs(a - b).mean()
    mx = np.abs(a - b).max()
    rms = np.sqrt(((a - b) ** 2).mean())
    # 图像域 [-1,1] 的 MAE; 转 [0,1] 域与 NEXT_ROUND 的 diff<0.01 口径一致
    mae01 = mae / 2.0
    print(f'[align] shape={out32.shape} MAE={mae01:.5f} (0-1域) MAX={mx:.5f} RMS={rms:.5f}', flush=True)
    verdict = 'PASS (diff<0.01, fp16 渲染可用于门控)' if mae01 < 0.01 else 'FAIL (diff>=0.01, 需排查)'
    print(f'[align] {verdict}', flush=True)
    with open(os.path.join(temp_args.output_dir, 'align_report.txt'), 'w') as f:
        f.write(f'MAE(0-1)={mae01:.6f} MAX={mx:.6f} RMS={rms:.6f}\n{verdict}\n')
    print(f'[done] total {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
