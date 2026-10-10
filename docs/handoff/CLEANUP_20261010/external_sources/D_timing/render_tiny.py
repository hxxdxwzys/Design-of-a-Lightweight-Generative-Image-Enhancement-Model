# -*- coding: utf-8 -*-
"""用微模型 ckpt 出 100 张 4K (分块+羽化), q98
用法: python render_tiny.py --ckpt <ckpt> --out D:/models/xxx/output_dir [--down 2]
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, r'D:/tmp_timing')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
TILE, OVER = 512, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--down', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
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
        if a.down > 1:
            t = F.interpolate(t, size=(TILE // a.down, TILE // a.down), mode='bicubic', align_corners=False)
        th = t.half()
        pred = net(th, torch.full((1,), 500, dtype=torch.long, device='cuda')).sample
        y = (th - sq_1a * pred.float()) / sq_a
        if a.down > 1:
            y = F.interpolate(y, size=(TILE, TILE), mode='bicubic', align_corners=False)
        o = y[0].permute(1, 2, 0).float().cpu().numpy()
        return np.clip((o + 1) * 127.5, 0, 255)

    files = sorted(TEST.glob('case*.jpg'), key=lambda p: int(p.stem[4:]))
    if a.limit:
        files = files[:a.limit]
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
        res = np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)
        Image.fromarray(res).save(out / p.name, quality=98, subsampling=0)
        if n % 20 == 0 or n == len(files):
            print(f'  {n}/{len(files)} {(time.time()-t0)/n:.2f}s/张', flush=True)
    print(f'[render] {len(files)} 张 -> {out} ({(time.time()-t0)/60:.1f} min)', flush=True)
    print('RENDER_TINY_DONE')


if __name__ == '__main__':
    main()
