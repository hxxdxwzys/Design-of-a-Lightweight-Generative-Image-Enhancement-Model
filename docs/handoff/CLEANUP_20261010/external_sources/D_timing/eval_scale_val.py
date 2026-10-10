# -*- coding: utf-8 -*-
"""补跑 val scale 扫描的评测 (2026-09-11): 渲染已全部完成, 只补 8 指标
用法: python eval_scale_val.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROXY = 'http://127.0.0.1:7890'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ[k] = PROXY
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PY = r'D:/Python/python.exe'
EVAL = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace/eval.py'
VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
WORK = Path(r'D:/models/sweep_scale_val')
OUTJ = Path(r'D:/tmp_timing/sweep_scale_val.json')

res = json.loads(OUTJ.read_text(encoding='utf-8')) if OUTJ.exists() else {}
scales = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
for sc in scales:
    tag = f'sc{sc}'
    pred = WORK / f'pred_{tag}'
    js = WORK / f'eval_{tag}.json'
    if tag in res:
        print(f'[skip] {tag} 已有'); continue
    if not pred.exists():
        print(f'[skip] {tag} 无渲染'); continue
    print(f'[评测] {tag} ...', flush=True)
    r = subprocess.run([PY, EVAL, '--pred-dir', str(pred), '--pred-pattern', '*_lq.*',
                        '--gt-dir', str(VAL), '--gt-pattern', '*_gt.*', '--out', str(js)],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=3600)
    if not js.exists():
        print(f'  !! 失败 rc={r.returncode}: {(r.stdout or "")[-300:]} {(r.stderr or "")[-500:]}')
        continue
    d = json.loads(js.read_text(encoding='utf-8'))
    res[tag] = d.get('mean', d)
    OUTJ.write_text(json.dumps(res, indent=1), encoding='utf-8')
    m = res[tag]
    print(f'  OK ' + ' '.join(f'{k}={m.get(k)}' for k in
                              ('psnr', 'ssim', 'lpips', 'dists', 'clipiqa', 'musiq', 'niqe', 'maniqa')))

print('\n=== val scale 扫描汇总 (3 张, 全 8 指标) ===')
keys = ['psnr', 'ssim', 'lpips', 'dists', 'clipiqa', 'musiq', 'niqe', 'maniqa']
print('  ' + 'scale'.ljust(7) + ''.join(k[:8].rjust(10) for k in keys))
for tag in sorted(res, key=lambda t: float(t[2:])):
    m = res[tag]
    row = tag.ljust(7)
    for k in keys:
        v = m.get(k)
        row += ('     -    ' if v is None else f'{v:10.4f}')
    print('  ' + row)
print(f'\n已存 {OUTJ}')
