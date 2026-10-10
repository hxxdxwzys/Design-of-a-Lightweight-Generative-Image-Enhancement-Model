# -*- coding: utf-8 -*-
"""2x 真细节探针 (2026-09-13 晚): 同脚本同种子, 1x vs 2x->降采样 的细节增益
用途: 判断"真细节"(不是后处理锐化)能不能把 lapvar 抬起来, 若能 -> 100 张全量重出图
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
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
ROOT = Path(r'D:/tmp_timing/x2probe'); ROOT.mkdir(exist_ok=True)
CASES = [2, 13, 50]
SEED = 8

def lapvar_gray(g):
    return float(ndimage.laplace(g).var())

def run(cmd, log):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env, timeout=3600)
    with log.open('a', encoding='utf-8') as f:
        f.write((r.stdout or '') + (r.stderr or '') + '\n')
    return r.returncode, time.time() - t0

res = {}
for c in CASES:
    src = TEST / f'case{c}.jpg'
    orig = Image.open(src).convert('RGB')
    W, H = orig.size
    for tag, up in (('1x', 1), ('2x', 2)):
        outd = ROOT / tag
        outd.mkdir(exist_ok=True)
        png = outd / f'case{c}.png'
        if not png.exists():
            rc, dt = run([PY, INFER, '-c', BASE, '--lora', LORA, '--lora-scale', '0.8',
                          '--seed', str(SEED), '-u', str(up), '-i', str(src), '-o', str(outd),
                          '--align_method', 'wavelet'], ROOT / f'{tag}.log')
            print(f'[{tag}] case{c} rc={rc} {dt:.0f}s', flush=True)
            if rc != 0:
                continue
        im = Image.open(png).convert('RGB')
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)     # 2x 输出降回原尺寸
        jpg = ROOT / f'{tag}_case{c}.jpg'
        im.save(jpg, quality=98, subsampling=0)
        lv = lapvar_gray(np.asarray(im.convert('L'), dtype=np.float32))
        res[(c, tag)] = lv
        if c == CASES[0]:
            im.crop((1500, 1200, 1500+512, 1200+512)).save(ROOT / f'{tag}_crop.png')
        print(f'  case{c} {tag}: size={im.size} lapvar={lv:.1f}', flush=True)

print('\n=== 汇总 (同种子同脚本) ===')
for c in CASES:
    a, b = res.get((c, '1x')), res.get((c, '2x'))
    if a and b:
        print(f'case{c}: 1x={a:.1f}  2x->ds={b:.1f}  比值={b/a:.2f}')
print('X2PROBE_DONE')
