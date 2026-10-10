# -*- coding: utf-8 -*-
"""探针 A 预验证 (2026-09-11): 在 30 张测试集子集上渲 scale=0.4, 与 M24(sc0.8) 直证对比
对比基线: M24 在同 30 张(same sorted()[:30])上的 niqe=5.2559 / clipiqa=0.9023 (nr_test100_results.json)
若 sc0.4 的测试集 niqe 显著更低 -> 全量重渲 100 张并提交
用法: python probe_a_screen.py [--scale 0.4] [--n 30]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROXY = 'http://127.0.0.1:7890'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ[k] = PROXY
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PY = r'D:/Python/python.exe'
INFER = r'D:/tmp_timing/vosr_lora_infer.py'
BASE = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
WORK = Path(r'D:/models/probe_a')
OUT = Path(r'D:/tmp_timing/probe_a_screen.json')
BASELINE = {'M24_sc0.8': {'niqe': 5.2559, 'clipiqa': 0.9023, 'musiq': 46.1257}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=float, default=0.4)
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    sub = WORK / f'lq{a.n}'
    sub.mkdir(parents=True, exist_ok=True)
    files = sorted(TEST.glob('*.jpg'))[:a.n]
    for f in files:
        d = sub / f.name
        if not d.exists():
            shutil.copy2(f, d)
    print(f'[subset] {a.n} 张 (与 M24 基线同口径 sorted()[:{a.n}])', flush=True)

    tag = f'sc{a.scale}'
    outd = WORK / f'pred_{tag}'
    print(f'[渲染] {tag} ...', flush=True)
    t0 = time.time()
    r = subprocess.run(
        f'{PY} {INFER} -c {BASE} --lora {LORA} --lora-scale {a.scale} --seed 8 '
        f'--align_method wavelet -i {sub} -o {outd}',
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=10800, cwd=r'C:/models/finals')
    if r.returncode != 0:
        print(f'[渲染失败] {(r.stderr or "")[-400:]}'); return
    print(f'  渲染完成 ({time.time()-t0:.0f}s, {len(list(outd.glob("*.jpg")))} 张)', flush=True)

    import numpy as np
    import torch
    import pyiqa
    from PIL import Image
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = {k: pyiqa.create_metric(k, device=dev) for k in ('niqe', 'clipiqa', 'musiq')}
    vals = {k: [] for k in metrics}
    t0 = time.time()
    for j in sorted(outd.glob('*.jpg')):
        im = Image.open(j).convert('RGB')
        t = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        for k, m in metrics.items():
            with torch.no_grad():
                vals[k].append(float(m(t).item()))
        if len(vals['niqe']) % 10 == 0:
            print(f'  指标 {len(vals["niqe"])}/{a.n} ({time.time()-t0:.0f}s)', flush=True)
    res = {k: round(float(np.mean(v)), 4) for k, v in vals.items()}
    b = BASELINE['M24_sc0.8']
    print(f'\n=== 测试集 {a.n} 张直证对比 ===')
    print(f'  {"":<12}{"niqe":>9}{"clipiqa":>10}{"musiq":>9}')
    print(f'  {"M24 (sc0.8)":<12}{b["niqe"]:>9.4f}{b["clipiqa"]:>10.4f}{b["musiq"]:>9.4f}')
    print(f'  {tag:<12}{res["niqe"]:>9.4f}{res["clipiqa"]:>10.4f}{res["musiq"]:>9.4f}')
    print(f'  {"Δ":<12}{res["niqe"]-b["niqe"]:>+9.4f}{res["clipiqa"]-b["clipiqa"]:>+10.4f}'
          f'{res["musiq"]-b["musiq"]:>+9.4f}')
    pred = 81.498 + 0.245 * res['musiq'] - 9.449 * res['niqe']
    pred0 = 81.498 + 0.245 * b['musiq'] - 9.449 * b['niqe']
    print(f'\n  2 参数模型预测: M24={pred0:.2f} -> {tag}={pred:.2f}  (Δ={pred-pred0:+.2f})')
    print(f'  （M24 实测官方分 43.6951; 模型在该点偏高 {pred0-43.6951:+.2f})')
    OUT.write_text(json.dumps({'scale': a.scale, 'n': a.n, 'metrics': res,
                               'baseline_M24': b, 'pred': round(pred, 3)}, indent=1),
                   encoding='utf-8')
    print(f'\n已存 {OUT}')


if __name__ == '__main__':
    main()
