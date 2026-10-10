# -*- coding: utf-8 -*-
"""微模型评估: val 3 张 4K 全图分块推理 -> 与 M24 同口径比 PSNR/SSIM/LPIPS + 512 时延
用法: python eval_tiny.py --ckpt D:/models/tiny_v1/v1b_s4000.pt [--out D:/models/tiny_v1/eval_s4000]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault('HF_HUB_OFFLINE', '1')
VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
TILE, OVER = 512, 32


def load_unet(ckpt, dev):
    from diffusers import UNet2DModel
    d = torch.load(ckpt, map_location='cpu', weights_only=False)
    ch = tuple(d['ch'])
    m = UNet2DModel(sample_size=d['res'], in_channels=3, out_channels=3, layers_per_block=1,
                    block_out_channels=ch,
                    down_block_types=('DownBlock2D',) * len(ch),
                    up_block_types=('UpBlock2D',) * len(ch))
    m.load_state_dict(d['unet']); m = m.to(dev).eval()
    return m, float(d['alpha']), d['res']


@torch.no_grad()
def run_tile(model, sq_a, sq_1a, x):
    pred = model(x, torch.full((x.shape[0],), 500, dtype=torch.long, device=x.device)).sample
    return (x - sq_1a * pred.float()) / sq_a


@torch.no_grad()
def enhance(model, sq_a, sq_1a, img_u8, down=1):
    """4K 分块推理, 羽化拼接"""
    H, W = img_u8.shape[:2]
    out = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)
    mask = np.ones((TILE, TILE, 1), dtype=np.float32)
    r = OVER // 2
    ramp = np.linspace(0, 1, r, dtype=np.float32)
    mask[:r, :, 0] *= ramp[:, None]; mask[-r:, :, 0] *= ramp[::-1, None]
    mask[:, :r, 0] *= ramp[None, :]; mask[:, -r:, 0] *= ramp[::-1][None, :]
    ys = list(range(0, max(1, H - TILE + 1), TILE - OVER)) + [max(0, H - TILE)]
    xs = list(range(0, max(1, W - TILE + 1), TILE - OVER)) + [max(0, W - TILE)]
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            patch = img_u8[y:y + TILE, x:x + TILE]
            t = torch.from_numpy(patch).permute(2, 0, 1)[None].to('cuda').float().div_(127.5).sub_(1)
            tin = t
            if down > 1:
                tin = F.interpolate(t, size=(TILE // down, TILE // down), mode='bicubic', align_corners=False)
            o_t = run_tile(model, sq_a, sq_1a, tin)
            if down > 1:
                o_t = F.interpolate(o_t, size=(TILE, TILE), mode='bicubic', align_corners=False)
            o = o_t[0].permute(1, 2, 0).float().cpu().numpy()
            o = np.clip((o + 1) * 127.5, 0, 255)
            out[y:y + TILE, x:x + TILE] += o * mask
            wsum[y:y + TILE, x:x + TILE] += mask
    return np.clip(out / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim_np(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(), b.var()
    cov = ((a - m1) * (b - m2)).mean()
    return ((2 * m1 * m2 + C1) * (2 * cov + C2)) / ((m1 ** 2 + m2 ** 2 + C1) * (v1 + v2 + C2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--down', type=int, default=1)
    ap.add_argument('--lpips', action='store_true', default=True)
    a = ap.parse_args()
    dev = 'cuda'
    model, alpha, res = load_unet(a.ckpt, dev)
    sq_a = alpha ** 0.5; sq_1a = (1 - alpha) ** 0.5
    print(f'[ckpt] {a.ckpt} ch/res={res} alpha={alpha:.4f}', flush=True)

    # 512 时延
    x = (torch.rand(1, 3, 512, 512, device=dev) * 2 - 1)
    with torch.no_grad():
        for _ in range(5):
            run_tile(model, sq_a, sq_1a, x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        tin = F.interpolate(x, size=(512 // a.down, 512 // a.down), mode='bicubic', align_corners=False) if a.down > 1 else x
        for _ in range(5):
            run_tile(model, sq_a, sq_1a, tin)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(50):
            run_tile(model, sq_a, sq_1a, tin)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 50 * 1000
    print(f'[latency] 512: {ms:.2f} ms -> 加速比 ~{285.6/ms:.1f}x', flush=True)

    import lpips as lpips_mod
    lp = lpips_mod.LPIPS(net='alex', verbose=False)
    outdir = Path(a.out) if a.out else Path(a.ckpt).parent / (Path(a.ckpt).stem + '_eval')
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in (1, 2, 3):
        lq_p = next(VAL.glob(f'case{c}_lq.*')); gt_p = next(VAL.glob(f'case{c}_gt.*'))
        lq = np.asarray(Image.open(lq_p).convert('RGB'), dtype=np.uint8)
        gt = np.asarray(Image.open(gt_p).convert('RGB'), dtype=np.uint8)
        pred = enhance(model, sq_a, sq_1a, lq, a.down)
        Image.fromarray(pred).save(outdir / f'case{c}_pred.png')
        H, W = pred.shape[:2]
        y0, x0 = H // 2 - 256, W // 2 - 256
        t = lambda z: torch.from_numpy(z[y0:y0+512, x0:x0+512].transpose(2, 0, 1)[None] / 127.5 - 1).float()
        with torch.no_grad():
            l_p = float(lp(t(pred), t(gt)).item()); l_g = float(lp(t(lq), t(gt)).item())
        rows.append((c, psnr(pred, gt), ssim_np(pred, gt), l_p, psnr(lq, gt), ssim_np(lq, gt), l_g))
        print(f'case{c}: PSNR {rows[-1][1]:.3f} (LQ {rows[-1][4]:.3f}) | SSIM {rows[-1][2]:.4f} (LQ {rows[-1][5]:.4f}) '
              f'| LPIPS512 {l_p:.4f} (LQ {l_g:.4f})', flush=True)
    m = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows])
    print(f'\n均值: PSNR {m[:,0].mean():.3f} (LQ {m[:,3].mean():.3f}) | SSIM {m[:,1].mean():.4f} (LQ {m[:,4].mean():.4f}) '
          f'| LPIPS {m[:,2].mean():.4f} (LQ {m[:,5].mean():.4f})')
    print(f'[参照 M24] PSNR 24.559 | SSIM 0.9658 | LPIPS 0.4408   [加速比参照] M24 0.780x')
    json.dump({'ckpt': a.ckpt, 'latency_ms': ms, 'speedup': 285.6 / ms,
               'psnr': float(m[:,0].mean()), 'ssim': float(m[:,1].mean()), 'lpips512': float(m[:,2].mean()),
               'lq_psnr': float(m[:,3].mean()), 'lq_ssim': float(m[:,4].mean()), 'lq_lpips': float(m[:,5].mean())},
              open(outdir / 'eval.json', 'w'), indent=1)
    print('EVAL_TINY_DONE')


if __name__ == '__main__':
    main()
