# -*- coding: utf-8 -*-
"""按与 measure_test30 完全相同的子集渲染 (2026-09-15)
measure_test30/measure_dir 用的是 sorted(TEST.glob('*.jpg'))[:30] (字典序: case1, case10, case100, case11...)
render_tiny 的 --limit 30 是数值序 (case1..case30) => 两者不重合, 数字不可比。
本脚本按字典序取前 N 张渲染, 保证与历史 test30 指标同口径。
用法: python render_subset.py --ckpt X.pt --out DIR --down 2 [--n 30] [--order lex|num]
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault('HF_HUB_OFFLINE', '1')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
TILE, OVER = 512, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--down', type=int, default=2)
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--order', default='lex')
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
        if a.down > 1:
            t = F.interpolate(t, size=(TILE // a.down, TILE // a.down), mode='bicubic', align_corners=False)
        th = t.half()
        pred = net(th, torch.full((1,), 500, dtype=torch.long, device='cuda')).sample
        y = (th - sq_1a * pred.float()) / sq_a
        if a.down > 1:
            y = F.interpolate(y, size=(TILE, TILE), mode='bicubic', align_corners=False)
        o = y[0].permute(1, 2, 0).float().cpu().numpy()
        return np.clip((o + 1) * 127.5, 0, 255)

    allf = list(TEST.glob('*.jpg'))
    files = sorted(allf)[:a.n] if a.order == 'lex' else sorted(allf, key=lambda p: int(p.stem[4:]))[:a.n]
    print('[render_subset] %d 张 (order=%s) -> %s' % (len(files), a.order, out), flush=True)
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
    print('[render_subset] done %.1f min' % ((time.time() - t0) / 60), flush=True)
    print('RENDER_SUBSET_DONE')


if __name__ == '__main__':
    main()
