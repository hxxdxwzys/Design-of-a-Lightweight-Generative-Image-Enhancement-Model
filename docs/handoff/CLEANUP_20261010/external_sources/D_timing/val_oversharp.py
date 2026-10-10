# -*- coding: utf-8 -*-
""""我们是否相对 GT 过锐?" 的直证 (2026-09-13)
val 有 GT -> 直接比 lapvar/hf: val_LQ / val_GT / 我们模型在 val 上的输出
若 输出 的细节量 >> GT -> 有参指标(psnr/ssim/lpips/dists)被我们自己惩罚
"""
from __future__ import annotations
import subprocess, sys, time, os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

PY = r'D:/Python/python.exe'
INFER = r'D:/tmp_timing/vosr_lora_infer.py'
BASE = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
OUT = Path(r'D:/tmp_timing/val_over_sharp'); OUT.mkdir(exist_ok=True)
LOG = OUT / 'run.log'

def feats(p):
    g = np.asarray(Image.open(p).convert('L'), dtype=np.float32)
    return (float(ndimage.laplace(g).var()),
            float(np.sqrt(np.mean((g - ndimage.gaussian_filter(g, 2.0)) ** 2))))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

rows = []
for c in (1, 2, 3):
    lq = next(VAL.glob(f'case{c}_lq.*'))
    gt = next(VAL.glob(f'case{c}_gt.*'))
    png = OUT / f'case{c}_pred.png'
    if not png.exists():
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        r = subprocess.run([PY, INFER, '-c', BASE, '--lora', LORA, '--lora-scale', '0.8',
                            '--seed', '8', '-u', '1', '-i', str(lq), '-o', str(OUT),
                            '--align_method', 'wavelet'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, timeout=1800)
        with LOG.open('a', encoding='utf-8') as f:
            f.write((r.stdout or '') + (r.stderr or '') + '\n')
        print(f'[render] case{c} rc={r.returncode}', flush=True)
    if not png.exists():
        print(f'  case{c}: 渲染失败'); continue
    # 渲染器可能输出 caseN_lq.png 之类名字, 兜底找
    cand = sorted(OUT.glob(f'case{c}*.png'))
    png = [p for p in cand if 'pred' in p.name] or cand
    png = png[0] if isinstance(png, list) else png
    a = feats(lq); b = feats(gt); d = feats(png)
    rows.append((c, a, b, d))
    print(f'case{c}: LQ lapvar={a[0]:7.1f} hf={a[1]:5.2f} | GT lapvar={b[0]:7.1f} hf={b[1]:5.2f} '
          f'| OUR lapvar={d[0]:8.1f} hf={d[1]:5.2f}  (OUR/GT lapvar={d[0]/b[0]:.2f})', flush=True)

if rows:
    lq = np.array([r[1] for r in rows]); gt = np.array([r[2] for r in rows]); ou = np.array([r[3] for r in rows])
    print(f'\n均值: LQ lapvar={lq[:,0].mean():.1f} | GT lapvar={gt[:,0].mean():.1f} | OUR lapvar={ou[:,0].mean():.1f} '
          f'(OUR/GT={ou[:,0].mean()/gt[:,0].mean():.2f})')
    print(f'      LQ hf={lq[:,1].mean():.2f} | GT hf={gt[:,1].mean():.2f} | OUR hf={ou[:,1].mean():.2f} '
          f'(OUR/GT={ou[:,1].mean()/gt[:,1].mean():.2f})')
print('VAL_OVERSHARP_DONE')
