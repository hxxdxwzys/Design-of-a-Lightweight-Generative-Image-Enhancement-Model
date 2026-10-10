# -*- coding: utf-8 -*-
"""噪声注入探针 (2026-09-15): 不训练, 只改推理 —— 给输入叠一点噪声再预测 x0
理论: 学生是确定性回归器 => 输出趋向"条件均值"(糊); 加噪后模型要"去掉自己不认识的那部分",
      相当于从更丰富的条件分布里采样 => 可能恢复教师那种纹理丰富度 (v7gan 轨迹显示 clipiqa 与纹理真实感同向)
最省成本的观感杠杆: 不改速度 (推理仍是 1 次前向), 不改权重。
用法: python probe_noise.py --ckpt D:/models/tiny_v1/v7gan_s25000.pt --out D:/models/probe_noise --sigmas 0.03,0.06,0.10 --n 10
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

os.environ.setdefault('HF_HUB_OFFLINE', '1')
for _k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(_k, None)
TEST = Path('C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
TILE, OVER = 512, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sigmas', default='0.03,0.06,0.10')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--down', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from diffusers import UNet2DModel
    import torch.nn.functional as F
    d = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    ch = tuple(d['ch']); res = int(d['res']); alpha = float(d['alpha'])
    unet = UNet2DModel(sample_size=res, in_channels=3, out_channels=3, layers_per_block=1,
                       block_out_channels=ch, down_block_types=('DownBlock2D',) * len(ch),
                       up_block_types=('UpBlock2D',) * len(ch))
    unet.load_state_dict(d['unet'])
    net = unet.to('cuda').eval().half()
    sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5
    mask = np.ones((TILE, TILE, 1), dtype=np.float32)
    r = OVER // 2
    ramp = np.linspace(0, 1, r, dtype=np.float32)
    mask[:r, :, 0] *= ramp[:, None]; mask[-r:, :, 0] *= ramp[::-1, None]
    mask[:, :r, 0] *= ramp[None, :]; mask[:, -r:, 0] *= ramp[::-1][None, :]

    @torch.no_grad()
    def run(patch_u8, sigma):
        t = torch.from_numpy(patch_u8).permute(2, 0, 1)[None].to('cuda').float().div_(127.5).sub_(1)
        if a.down > 1:
            t = F.interpolate(t, size=(TILE // a.down, TILE // a.down), mode='bicubic', align_corners=False)
        if sigma > 0:
            t = t + torch.randn_like(t) * sigma
        th = t.half()
        pred = net(th, torch.full((1,), 500, dtype=torch.long, device='cuda')).sample
        y = (th - sq_1a * pred.float()) / sq_a
        if a.down > 1:
            y = F.interpolate(y, size=(TILE, TILE), mode='bicubic', align_corners=False)
        o = y[0].permute(1, 2, 0).float().cpu().numpy()
        return np.clip((o + 1) * 127.5, 0, 255)

    files = sorted(TEST.glob('*.jpg'))[:a.n]
    sigmas = [float(s) for s in a.sigmas.split(',')]
    for sigma in sigmas:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        out = Path(a.out) / ('s%.3f' % sigma)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for p in files:
            img = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
            H, W = img.shape[:2]
            acc = np.zeros((H, W, 3), dtype=np.float32); wsum = np.zeros((H, W, 1), dtype=np.float32)
            ys = sorted(set(list(range(0, max(1, H - TILE + 1), TILE - OVER)) + [max(0, H - TILE)]))
            xs = sorted(set(list(range(0, max(1, W - TILE + 1), TILE - OVER)) + [max(0, W - TILE)]))
            for y in ys:
                for x in xs:
                    acc[y:y + TILE, x:x + TILE] += run(img[y:y + TILE, x:x + TILE], sigma) * mask
                    wsum[y:y + TILE, x:x + TILE] += mask
            res_img = np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)
            Image.fromarray(res_img).save(out / p.name, quality=98, subsampling=0)
        print('[noise] sigma=%.3f -> %s (%.1f min)' % (sigma, out, (time.time() - t0) / 60), flush=True)
    print('PROBE_NOISE_DONE')


if __name__ == '__main__':
    main()
