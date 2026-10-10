# -*- coding: utf-8 -*-
"""种子平均的指标测量 (2026-09-12 v2) — 走项目已验证的 eval.py, 强制 CPU
教训: 自写的 GPU 指标函数在原生 4K 上算 LPIPS/DISTS 会把 VGG 特征铺满 8GB 显存
      (实际申请 3.00 GiB 失败, 崩在 dists_arch.py)。项目自带的 eval.py 早就跑通过 4K,
      改用它可以保证数字与我们其它 val 口径完全一致, 且不占 GPU。
用法: python eval_seed_ensemble.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
EVAL = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace/eval.py'
VAL = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集')
WORK = Path(r'D:/models/seed_ensemble_test')
REF = {'psnr': 24.604, 'ssim': 0.7417, 'lpips': 0.3789, 'dists': 0.2960,
       'clipiqa': 0.9193, 'musiq': 48.141, 'niqa': None, 'niqe': 5.1723}

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def main():
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''          # 强制 CPU, 与 GPU 渲染并行不冲突
    env['HF_HUB_OFFLINE'] = '1'
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        env.pop(k, None)

    res = {}
    for tag in ('s8', 'avg4'):
        d = WORK / tag
        if not d.exists():
            print(f'[skip] {tag} 不存在'); continue
        out = T / f'seed_{tag}_eval.json'
        cmd = [PY, EVAL, '--pred-dir', str(d), '--pred-pattern', 'case*_lq.*',
               '--gt-dir', str(VAL), '--gt-pattern', 'case*_gt.*', '--out', str(out)]
        print(f'[eval] {tag} ...', flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', env=env, cwd=str(Path(EVAL).parent))
        print((r.stdout or '')[-400:], flush=True)
        if out.exists():
            res[tag] = json.loads(out.read_text(encoding='utf-8'))
    if len(res) < 2:
        print('[x] 数据不足'); return

    keys = ['psnr', 'ssim', 'lpips', 'dists', 'clipiqa', 'musiq', 'maniqa', 'niqe']
    print(f'\n=== 多种子平均 vs 单样本 (验证集 3 张, 原生 4K) ===')
    print(f'  {"指标":<9}{"单样本":>10}{"4种子平均":>12}{"Δ":>10}{"sc0.8 参照":>12}')
    a, b = res['s8'], res['avg4']
    for k in keys:
        if k in a and k in b:
            print(f'  {k:<9}{a[k]:>10.4f}{b[k]:>12.4f}{b[k]-a[k]:>+10.4f}{REF.get(k) or 0:>12.4f}')
    gain = REF['lpips'] - b.get('lpips', 9)
    print(f'\n  lpips: sc0.8 单样本 {REF["lpips"]} -> 4 种子平均 {b.get("lpips"):.4f} '
          f'({gain:+.4f})')
    print(f'  门槛判定: {"✅ 达到 <0.30, 阶段 2 值得花 12.7h" if b.get("lpips", 9) < 0.30 else "❌ 未达 0.30, 建议砍掉阶段 2"}')
    (T / 'seed_ensemble_val.json').write_text(
        json.dumps({'single': a, 'avg4': b, 'ref_sc08': REF}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    print(f'已存 {T / "seed_ensemble_val.json"}')


if __name__ == '__main__':
    main()
