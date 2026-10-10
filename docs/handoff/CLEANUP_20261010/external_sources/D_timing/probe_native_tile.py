# -*- coding: utf-8 -*-
"""原生分辨率探针 (2026-09-15): 不训练, 直接把 256 域 v7gan 权重用在【原生 256 tile】上
动机: v7gan 训练时见到的输入是 512 patch 降采样到 256 (down=2), 输出再升回 512 => 频带受限.
      若把 4K 图直接用 256 tile 原生推理 (无降采样), 学生能否产出更高频的"观感"? => 3 分钟探针
用法: python probe_native_tile.py --ckpt D:/models/tiny_v1/v7gan_s25000.pt --out D:/models/probe_native --n 30
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

os.environ.setdefault('HF_HUB_OFFLINE', '1')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
TILE, OVER = 256, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from diffusers import UNet2DModel
    d = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    ch = tuple(d['ch']); res = int(d['res']); alpha = float(d['alpha'])
    unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                       block_out_channels=ch, down_block_types=('DownBlock2D',) * len(ch),
                       up_block_types=('UpBlock2D',) * len(ch))
    unet.load_state_dict(d['unet'])
    net = unet.to('cuda').eval().half()
    sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mask = np.ones((TILE, TILE, 1), dtype=np.float32)
    r = OVER // 2
    ramp = np.linspace(0, 1, r, dtype=np.float32)
    mask[:r, :, 0] *= ramp[:, None]; mask[-r:, :, 0] *= ramp[::-1, None]
    mask[:, :r, 0] *= ramp[None, :]; mask[:, -r:, 0] *= ramp[::-1][None, :]

    @torch.no_grad()
    def run(patch_u8):
        t = torch.from_numpy(patch_u8).permute(2, 0, 1)[None].to('cuda').float().div_(127.5).sub_(1)
        th = t.half()
        pred = net(th, torch.full((1,), 500, dtype=torch.long, device='cuda')).sample
        y = (th - sq_1a * pred.float()) / sq_a
        o = y[0].permute(1, 2, 0).float().cpu().numpy()
        return np.clip((o + 1) * 127.5, 0, 255)

    files = sorted(TEST.glob('*.jpg'))[:a.n]      # 与 measure_test30 同口径 (字典序)
    print('[probe_native] model res=%d, tile=%d (原生, 无降采样), %d 张' % (res, TILE, len(files)), flush=True)
    t0 = time.time()
    for n, p in enumerate(files, 1):
        img = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
        H, W = img.shape[:2]
        acc = np.zeros((H, W, 3), dtype=np.float32); wsum = np.zeros((H, W, 1), dtype=np.float32)
        ys = sorted(set(list(range(0, max(1, H - TILE + 1), TILE - OVER)) + [max(0, H - TILE)]))
        xs = sorted(set(list(range(0, max(1, W - TILE + 1), TILE - OVER)) + [max(0, W - TILE)]))
        for y in ys:
            for x in xs:
                acc[y:y + TILE, x:x + TILE] += run(img[y:y + TILE, x:x + TILE]) * mask
                wsum[y:y + TILE, x:x + TILE] += mask
        res_img = np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)
        Image.fromarray(res_img).save(out / p.name, quality=98, subsampling=0)
        if n % 10 == 0:
            print('  %d/%d %.1f min' % (n, len(files), (time.time() - t0) / 60), flush=True)
    print('[probe_native] done %.1f min -> %s' % ((time.time() - t0) / 60, out), flush=True)
    print('PROBE_NATIVE_DONE')


if __name__ == '__main__':
    main()
