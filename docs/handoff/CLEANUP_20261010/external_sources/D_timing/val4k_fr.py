# -*- coding: utf-8 -*-
"""val 4K 全图有参四项 (2026-09-15): 补上"官方口径"的盲区
动机: 我们所有有参比较都用 512 中心裁切; 而官方指标算在**完整 4K 出图**上(要分块拼接)。
      若某些模型的分块拼接有伪影/边界效应, 512 裁切口径会完全看不到 => 可能解释 PixB=18.97 与榜单 124+ 的差距。
做法: 用【已提交包的 runner 原样】对 3 张 val LQ 做 512 分块拼接 (overlap32 羽化), 再算 4K 四项。
用法: python val4k_fr.py [--only lq,m24,pixb,v9b,v11b]
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
for _k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(_k, None)
import numpy as np
import torch
from PIL import Image

DEV = 'cuda'
TILE, OVER = 512, 32
VAL = Path('C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
OUT = Path('D:/models/val4k')
OUT.mkdir(parents=True, exist_ok=True)
RES = Path('D:/tmp_timing/val4k_fr.json')

PKGS = {
    'm24':  ('D:/models/final_compliant/model_dir', 'M24 (0.5B, 官方 43.6951)'),
    'pixb': ('D:/models/pkg_PixB/model_dir', 'PixB (官方 18.9710)'),
    'v9b':  ('D:/models/pkg_TinyV9b/model_dir', 'TinyV9b (待交, 12.4x)'),
    'v7g':  ('D:/models/pkg_TinyV7gan/model_dir', 'TinyV7gan (官方 41.7346)'),
}
ap = argparse.ArgumentParser()
ap.add_argument('--only', default='lq,m24,pixb,v9b,v7g')
a = ap.parse_args()
todo = [x.strip() for x in a.only.split(',') if x.strip()]


def log(*xs):
    print(*xs, flush=True)


def psnr(p, g):
    m = np.mean((p.astype(np.float64) - g.astype(np.float64)) ** 2)
    return 99.0 if m <= 0 else float(10 * np.log10(255.0 ** 2 / m))


def ssim_np(p, g):
    p = p.astype(np.float64); g = g.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m1, m2 = p.mean(), g.mean(); v1, v2 = p.var(), g.var()
    cov = ((p - m1) * (g - m2)).mean()
    return float(((2 * m1 * m2 + C1) * (2 * cov + C2)) / ((m1 ** 2 + m2 ** 2 + C1) * (v1 + v2 + C2)))


import lpips as lpips_mod
import pyiqa
LP = lpips_mod.LPIPS(net='alex', verbose=False).to(DEV).eval()
DISTS = pyiqa.create_metric('dists', device=DEV)
to_t = lambda u8: torch.from_numpy(u8.copy()).permute(2, 0, 1)[None].to(DEV).float().div_(127.5).sub_(1)


def metrics4k(pred, gt):
    p, g = to_t(pred), to_t(gt)
    with torch.no_grad():
        l = float(LP(p, g).item())
        d = float(DISTS((p + 1) / 2, (g + 1) / 2).item())
    return psnr(pred, gt), ssim_np(pred, gt), l, d


def make_runner(key):
    mdir, _ = PKGS[key]
    spec = importlib.util.spec_from_file_location('r_' + key, mdir + '/runner.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['r_' + key] = mod
    sys.path.insert(0, mdir)
    spec.loader.exec_module(mod)
    return mod.Runner(mdir)


mask = np.ones((TILE, TILE, 1), dtype=np.float32)
r = OVER // 2
ramp = np.linspace(0, 1, r, dtype=np.float32)
mask[:r, :, 0] *= ramp[:, None]; mask[-r:, :, 0] *= ramp[::-1, None]
mask[:, :r, 0] *= ramp[None, :]; mask[:, -r:, 0] *= ramp[::-1][None, :]


def render4k(runner, lq_u8):
    H, W = lq_u8.shape[:2]
    acc = np.zeros((H, W, 3), dtype=np.float32); wsum = np.zeros((H, W, 1), dtype=np.float32)
    ys = sorted(set(list(range(0, max(1, H - TILE + 1), TILE - OVER)) + [max(0, H - TILE)]))
    xs = sorted(set(list(range(0, max(1, W - TILE + 1), TILE - OVER)) + [max(0, W - TILE)]))
    for y in ys:
        for x in xs:
            t = torch.from_numpy(lq_u8[y:y + TILE, x:x + TILE].copy()).permute(2, 0, 1)[None].to(DEV).float().div_(127.5).sub_(1)
            with torch.no_grad():
                o = runner.infer(t)[0].permute(1, 2, 0).float().clamp(-1, 1).cpu().numpy()
            acc[y:y + TILE, x:x + TILE] += np.clip((o + 1) * 127.5, 0, 255) * mask
            wsum[y:y + TILE, x:x + TILE] += mask
    return np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)


res = json.loads(RES.read_text(encoding='utf-8')) if RES.exists() else {}
for key in todo:
    if key in res:
        log('[skip]', key)
        continue
    t0 = time.time()
    runner = None if key == 'lq' else make_runner(key)
    per = []
    for c in range(1, 4):
        lq = np.asarray(Image.open(next(VAL.glob('case%d_lq.*' % c))).convert('RGB'), dtype=np.uint8)
        gt = np.asarray(Image.open(next(VAL.glob('case%d_gt.*' % c))).convert('RGB'), dtype=np.uint8)
        pred = lq if runner is None else render4k(runner, lq)
        if runner is not None:
            Image.fromarray(pred).save(OUT / ('%s_case%d.jpg' % (key, c)), quality=95)
        p, s, l, d = metrics4k(pred, gt)
        per.append({'psnr': p, 'ssim': s, 'lpips': l, 'dists': d})
        log('  %s case%d PSNR %.3f SSIM %.4f LPIPS %.4f DISTS %.4f' % (key, c, p, s, l, d))
    agg = {k: float(np.mean([q[k] for q in per])) for k in per[0]}
    res[key] = dict(per_case=per, **agg, elapsed_s=round(time.time() - t0, 1))
    RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    log('[done] %-5s PSNR %.3f SSIM %.4f LPIPS %.4f DISTS %.4f (%.0fs)' % (key, agg['psnr'], agg['ssim'], agg['lpips'], agg['dists'], time.time() - t0))

log('')
log('| 配置 | PSNR | SSIM | LPIPS | DISTS |')
log('|---|---|---|---|---|')
for k in ['lq'] + list(PKGS.keys()):
    if k in res:
        v = res[k]
        log('| %s | %.3f | %.4f | %.4f | %.4f |' % (k, v['psnr'], v['ssim'], v['lpips'], v['dists']))
log('VAL4K_FR_DONE')
