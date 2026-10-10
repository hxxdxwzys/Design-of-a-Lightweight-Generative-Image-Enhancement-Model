# -*- coding: utf-8 -*-
"""PIPE-01 入口之二：离线文件入口 —— 读图 -> 同一个核心 infer -> 写 PNG。

薄入口：不另做增强/采样/颜色处理；写图是唯一的额外步骤（编码器与 tensor 入口共用）。
用法:
    D:/Python/python.exe offline.py --input <img or dir> --out <dir> [--config <json>] [--save-tensor <dir>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import core


def read_image_tensor(path, upscale=1):
    """读图 -> [1,3,H,W] float32 [-1,1]。upscale=1 时不做任何 resize。"""
    from PIL import Image
    im = Image.open(str(path)).convert('RGB')
    w, h = im.size
    if int(upscale) != 1:
        im = im.resize((w * int(upscale), h * int(upscale)), Image.BICUBIC)
    return core.pil_to_tensor(im), (w, h)


def process_image(assets, img_path, out_dir, save_tensor_dir=None):
    """离线入口的单图处理；返回可落盘记录（含写图前张量与 PNG 的哈希）。"""
    import hashlib
    import numpy as np
    img_path = Path(img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x, orig_wh = read_image_tensor(img_path, assets.cfg['upscale'])
    out = core.infer_core(assets, x)
    pil = core.tensor_to_pil(out)          # 与 tensor 入口共用的唯一编码器
    png_path = out_dir / (img_path.stem + '.png')
    pil.save(png_path)
    rec = {
        'entry': 'offline_file',
        'input_path': str(img_path),
        'input_wh': list(orig_wh),
        'tensor_shape': list(out.shape),
        'tensor_dtype': str(out.dtype),
        'tensor_min': float(out.min()), 'tensor_max': float(out.max()),
        'png_path': str(png_path),
        'png_bytes': png_path.stat().st_size,
        'png_sha256': core.sha256_file(png_path),
        'tensor_sha256': hashlib.sha256(out.detach().float().cpu().numpy().tobytes()).hexdigest(),
    }
    if save_tensor_dir:
        td = Path(save_tensor_dir)
        td.mkdir(parents=True, exist_ok=True)
        npy = td / (img_path.stem + '.npy')
        np.save(npy, out.detach().float().cpu().numpy())
        rec['tensor_npy'] = str(npy)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description='PIPE-01 离线文件入口（读图 -> 共享核心 -> PNG）')
    ap.add_argument('--input', required=True, help='图片路径或目录')
    ap.add_argument('--out', required=True, help='PNG 输出目录')
    ap.add_argument('--config', default=None, help='冻结配置路径（默认包内 config.json）')
    ap.add_argument('--save-tensor', default=None, help='可选：写图前张量的 .npy 目录')
    ap.add_argument('--model-dir', default=None, help='仅记录用途（本阶段资产来自冻结配置）')
    args = ap.parse_args(argv)

    cfg = core.load_config(args.config)          # 纯 CPU：不加载模型、不启动 CUDA
    print('[cfg] sha256=%s tile=%s' % (cfg['_config_sha256'][:16], core.tile_config_report(cfg)))
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[dev] %s' % device, flush=True)
    assets = core.Assets(cfg, device)
    print('[load] %s' % assets.load_report, flush=True)
    print('[fp] model=%s' % assets.effective_fp['model']['digest_sha256'], flush=True)

    inp = Path(args.input)
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    if inp.is_dir():
        paths = sorted(p for p in inp.iterdir() if p.suffix.lower() in exts)
    else:
        paths = [inp]
    if not paths:
        raise SystemExit('没有找到输入图片: %s' % inp)
    recs = []
    for p in paths:
        rec = process_image(assets, p, args.out, args.save_tensor)
        recs.append(rec)
        print('[out] %s' % rec, flush=True)
    print('[done] %d 张 -> %s' % (len(recs), args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
