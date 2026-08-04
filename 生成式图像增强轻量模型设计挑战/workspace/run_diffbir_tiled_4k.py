#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DiffBIR 外部 512 tile 分块 4K 推理 (高斯融合拼接)。

绕开 DiffBIR 内置 tiled 的慢/卡问题: 每块独立跑 pipeline.run (512, 10步 ~19s),
块间高斯融合拼接。8GB 显存安全。

用法:
  python run_diffbir_tiled_4k.py <input_dir> <output_dir> [--tile 512] [--overlap 64] [--steps 10]
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
from PIL import Image

DIFFBIR_ROOT = 'D:/models/diffbir_code/DiffBIR-main/DiffBIR-main'
sys.path.insert(0, DIFFBIR_ROOT)
os.chdir(DIFFBIR_ROOT)  # config 用相对路径加载, 必须切到仓库根


def gaussian_weights(tile, overlap):
    w = np.ones((tile, tile), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap + 2)[1:-1]
        for i in range(overlap):
            v = ramp[i]
            w[i, :] = np.minimum(w[i, :], v)
            w[-1 - i, :] = np.minimum(w[-1 - i, :], v)
            w[:, i] = np.minimum(w[:, i], v)
            w[:, -1 - i] = np.minimum(w[:, -1 - i], v)
    return torch.from_numpy(w)


def build_loop(args):
    """用官方 parse_args 构造 BIDInferenceLoop (denoise + v2.1 + captioner none), 只加载不 run。"""
    import warnings
    warnings.filterwarnings('ignore')
    from accelerate.utils import set_seed
    from diffbir.inference import BIDInferenceLoop
    from inference import parse_args as diffbir_parse_args

    _old_argv = sys.argv
    sys.argv = ['inference.py', '--input', 'dummy', '--output', 'dummy']
    a = diffbir_parse_args()
    a.s_churn = 0
    a.s_noise = 0.0
    sys.argv = _old_argv
    # 覆盖为我们的配置
    a.task = 'denoise'
    a.upscale = 1
    a.version = 'v2.1'
    a.sampler = 'edm_dpm++_3m_sde'
    a.steps = args.steps
    a.start_point_type = 'cond'
    a.captioner = 'none'
    a.pos_prompt = ''
    a.neg_prompt = ''
    a.device = 'cuda'
    a.precision = 'fp16'
    a.seed = 42
    set_seed(a.seed)
    loop = BIDInferenceLoop(a)
    loop.setup = lambda: None  # 不建输出目录
    loop.load_cldm()
    loop.load_cond_fn()
    loop.load_pipeline()
    loop.load_captioner()
    return loop


def infer_tile(loop, lq_np: np.ndarray) -> np.ndarray:
    """lq_np [H,W,3] 0-255 -> 增强结果 [H,W,3] 0-255。"""
    import torch as _t
    pos_prompt = loop.args.pos_prompt
    neg_prompt = loop.args.neg_prompt
    with _t.autocast(loop.args.device, _t.float16 if loop.args.precision == 'fp16' else _t.float32):
        batch_samples = loop.pipeline.run(
            lq_np[None], loop.args.steps, loop.args.strength,
            loop.args.cleaner_tiled, loop.args.cleaner_tile_size, loop.args.cleaner_tile_stride,
            loop.args.vae_encoder_tiled, loop.args.vae_encoder_tile_size,
            loop.args.vae_decoder_tiled, loop.args.vae_decoder_tile_size,
            loop.args.cldm_tiled, loop.args.cldm_tile_size, loop.args.cldm_tile_stride,
            pos_prompt, neg_prompt, loop.args.cfg_scale, loop.args.start_point_type,
            loop.args.sampler, loop.args.noise_aug, loop.args.rescale_cfg,
            loop.args.s_churn, loop.args.s_tmin, loop.args.s_tmax, loop.args.s_noise,
            loop.args.eta, loop.args.order,
        )
    return np.clip(np.array(batch_samples[0]) * 255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_dir')
    ap.add_argument('output_dir')
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=64)
    ap.add_argument('--steps', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    loop = build_loop(args)
    print('DiffBIR 加载完成', flush=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    for i, f in enumerate(files):
        img = Image.open(f).convert('RGB')
        W, H = img.size
        print(f'[{i+1}/{len(files)}] {os.path.basename(f)} {W}x{H}', flush=True)
        arr = np.asarray(img, dtype=np.float32)
        tile, ov = args.tile, args.overlap
        pw = (tile - W % tile) % tile
        ph = (tile - H % tile) % tile
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode='reflect')
        H2, W2 = arr.shape[:2]
        out = np.zeros((H2, W2, 3), np.float32)
        wsum = np.zeros((H2, W2, 1), np.float32)
        g = gaussian_weights(tile, ov).numpy()[:, :, None]
        step = tile - ov
        for y in range(0, H2, step):
            for x in range(0, W2, step):
                y1 = min(y, H2 - tile)
                x1 = min(x, W2 - tile)
                patch = arr[y1:y1 + tile, x1:x1 + tile]
                enh = infer_tile(loop, patch).astype(np.float32)
                out[y1:y1 + tile, x1:x1 + tile] += enh * g
                wsum[y1:y1 + tile, x1:x1 + tile] += g
                print(f'  tile {y1},{x1} done', flush=True)
        out = out / wsum.clamp(min=1e-8) if hasattr(wsum, 'clamp') else out / np.maximum(wsum, 1e-8)
        out_np = np.clip(out[:H, :W], 0, 255).astype(np.uint8)
        Image.fromarray(out_np).save(os.path.join(args.output_dir, os.path.basename(f).rsplit('.', 1)[0] + '.jpg'), quality=95)
        print(f'  saved {os.path.basename(f).rsplit(chr(46), 1)[0]}.jpg', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
