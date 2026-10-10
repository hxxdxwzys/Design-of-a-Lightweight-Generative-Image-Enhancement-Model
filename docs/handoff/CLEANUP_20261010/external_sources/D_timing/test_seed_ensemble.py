# -*- coding: utf-8 -*-
"""免费验证 1 (2026-09-12): 多种子平均能否真的提升保真?
=========================================================
原理: 模型输出 = f(lq, z), z = randn_like(latent)。对 N 个种子的输出求平均 ≈ 条件均值,
      是 MSE 意义下更优的估计 -> 预期 psnr/ssim/lpips/dists 同时改善 (生成噪声方差下降)。
      这属于"同模型 + 更多离线推理", 出图不计入测速, 完全合法。

决策价值: 若 val lpips 从 0.379 (sc0.8 单样本) 压到 <0.30, 阶段 2 (12.7h 全量渲染 + 1 发) 才值得。
用法: python test_seed_ensemble.py [--seeds 8,108,208,308] [--scale 0.8]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
INFER = r'D:/tmp_timing/vosr_lora_infer.py'
BASE = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
WORK = Path(r'D:/models/seed_ensemble_test')
OUTJ = T / 'seed_ensemble_val.json'
REF = {'lpips': 0.3789, 'dists': 0.2960, 'psnr': 24.604, 'ssim': 0.7417,
       'niqe': 5.1723, 'musiq': 48.141, 'clipiqa': 0.9193}   # sc0.8 单样本 (sweep_scale_val.json)


def run(cmd, timeout=3600):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=timeout, cwd=r'C:/models/finals')
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def metrics(pred_dir: Path):
    """7 指标 vs 验证集 GT (4K 原生; 逐图匹配: case1_gt -> 预测 case1_lq)"""
    import numpy as np
    import torch
    import pyiqa
    from PIL import Image
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    pairs = []
    for g in sorted(VAL.glob('*_gt.*')):
        stem = g.name.split('_gt')[0]          # case1
        for cand in (f'{stem}_lq.png', f'{stem}_lq.jpg', f'{stem}.png', f'{stem}.jpg'):
            p = pred_dir / cand
            if p.exists():
                pairs.append((p, g)); break
    if not pairs:
        return None, 0
    ms = {k: pyiqa.create_metric(k, device=dev) for k in ('niqe', 'musiq', 'clipiqa')}
    lp = pyiqa.create_metric('lpips', device=dev)
    ds = pyiqa.create_metric('dists', device=dev)
    acc = {k: [] for k in ('psnr', 'ssim', 'lpips', 'dists', 'niqe', 'musiq', 'clipiqa')}
    for p, g in pairs:
        pi = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
        gi = np.asarray(Image.open(g).convert('RGB')).astype(np.float32) / 255.0
        if pi.shape != gi.shape:
            pi = np.asarray(Image.open(p).convert('RGB').resize(
                (gi.shape[1], gi.shape[0]), Image.BICUBIC)).astype(np.float32) / 255.0
        mse = float(((pi - gi) ** 2).mean())
        acc['psnr'].append(10 * np.log10(1.0 / max(mse, 1e-12)))
        mu_p, mu_g = pi.mean(), gi.mean()
        vp, vg = pi.var(), gi.var()
        cov = float(((pi - mu_p) * (gi - mu_g)).mean())
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        acc['ssim'].append(((2 * mu_p * mu_g + c1) * (2 * cov + c2)) /
                           ((mu_p ** 2 + mu_g ** 2 + c1) * (vp + vg + c2)))
        tp = torch.from_numpy(pi).permute(2, 0, 1).unsqueeze(0).to(dev)
        tg = torch.from_numpy(gi).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            acc['lpips'].append(float(lp(tp, tg).item()))
            acc['dists'].append(float(ds(tp, tg).item()))
            for k, m in ms.items():
                acc[k].append(float(m(tp).item()))
    return {k: round(float(np.mean(v)), 4) for k, v in acc.items()}, len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='8,108,208,308')
    ap.add_argument('--scale', type=float, default=0.8)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    WORK.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from PIL import Image
    import shutil

    # 只把 *_lq.* 放进独立输入目录 —— list_lq_images 会吃目录下所有图片,
    # 直接传验证集目录会把 _gt 也当输入渲染 (每种子 6 张而非 3 张, 且输出名对不上 GT)
    lqdir = WORK / 'lq'
    if not lqdir.exists():
        lqdir.mkdir(parents=True, exist_ok=True)
    for g in sorted(VAL.glob('*_lq.*')):
        d = lqdir / g.name
        if not d.exists():
            shutil.copy2(g, d)
    print(f'[input] LQ 输入 {len(list(lqdir.iterdir()))} 张 -> {lqdir}', flush=True)

    outs = []
    for sd in seeds:
        d = WORK / f's{sd}'
        d.mkdir(parents=True, exist_ok=True)
        if len(list(d.glob('*.png'))) < 3:
            for old in d.glob('*.png'):      # 清掉可能残缺的上一轮
                old.unlink()
            t0 = time.time()
            rc, out = run([PY, INFER, '-c', BASE, '--lora', LORA, '--lora-scale', str(a.scale),
                           '--seed', str(sd), '--align_method', 'wavelet',
                           '-i', str(lqdir), '-o', str(d)])
            print(f'[seed {sd}] rc={rc}, {len(list(d.glob("*.png")))} 张, {time.time()-t0:.0f}s', flush=True)
            if rc != 0:
                print('   ' + out[-300:]); continue
        outs.append(d)
    if len(outs) < 2:
        print('[x] 有效种子不足'); return

    # 逐像素平均
    avg = WORK / f'avg{len(outs)}'
    avg.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in outs[0].glob('*.png'))
    for n in names:
        arrs = [np.asarray(Image.open(d / n).convert('RGB')).astype(np.float32) for d in outs
                if (d / n).exists()]
        m = np.mean(arrs, axis=0)
        Image.fromarray(np.clip(m + 0.5, 0, 255).astype(np.uint8)).save(avg / n)

    print(f'\n=== 多种子平均 (scale={a.scale}, N={len(outs)}) ===', flush=True)
    res = {}
    for tag, d in [('single', outs[0]), (f'avg{len(outs)}', avg)]:
        m, npair = metrics(d)
        res[tag] = m
        print(f'  [{tag}] ({npair} 对) {m}', flush=True)

    s, av = res['single'], res[f'avg{len(outs)}']
    print(f'\n  {"指标":<9}{"单样本":>10}{f"平均({len(outs)})":>12}{"Δ":>10}{"sc0.8官方口径":>14}')
    for k in ('psnr', 'ssim', 'lpips', 'dists', 'niqe', 'musiq', 'clipiqa'):
        if k in s and k in av:
            print(f'  {k:<9}{s[k]:>10.4f}{av[k]:>12.4f}{av[k]-s[k]:>+10.4f}{REF.get(k, 0):>14.4f}')
    gain_lp = REF['lpips'] - av.get('lpips', REF['lpips'])
    print(f'\n  相对 sc0.8 单样本: lpips {gain_lp:+.4f} -> '
          f'{"✅ 值得全量渲染" if av.get("lpips", 9) < 0.30 else "❌ 未达 0.30 门槛, 建议砍掉阶段 2"}')
    OUTJ.write_text(json.dumps({'seeds': seeds, 'scale': a.scale, 'res': res,
                                'ref_sc08_single': REF}, indent=1), encoding='utf-8')
    print(f'已存 {OUTJ}')


if __name__ == '__main__':
    main()
