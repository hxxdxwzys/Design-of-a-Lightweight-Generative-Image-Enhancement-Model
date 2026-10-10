# -*- coding: utf-8 -*-
"""P3A 多指标画像探针 (2026-08-24): 4 源 x 8 指标 (PSNR/SSIM/LPIPS/DISTS/CLIPIQA/MUSIQ/MANIQA/NIQE)
源: v7_seed8(M24点, 需重渲染) / os base / ms base / 1.4B os (现有 val 输出)
问题: 官方公式 8 指标里我们只跟踪了 2 个 - musiq/maniqa/dists 等轴是否有被牺牲的源?
结果: D:/tmp_timing/p3a_*.json
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
TT = Path(r'D:/tmp_timing')
EVAL = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace/eval.py'
GT_DIR = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
VAL_IN = TT / 'codiff_val_in'

SOURCES = [
    ('v7_seed8_M24', r'D:/models/vosr_lora/v7/val_out_v7s5000_sc08_seed8_wavelet', True),
    ('os_base', r'C:/models/finals/preset/vosr_val_out', False),
    ('ms_base_cfg05', r'D:/models/vosr_ms/val_out_s8_cfg0.5', False),
    ('os_14b', r'C:/models/finals/preset/vosr_14b_val_out', False),
]


def gpu_free():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    # 1) re-render v7 seed8 val (M24 point)
    v7out = Path(r'D:/models/vosr_lora/v7/val_out_v7s5000_sc08_seed8_wavelet')
    if not v7out.exists() or len(list(v7out.glob('*.png'))) < 3:
        print('[probe] render v7 seed8 val...', flush=True)
        for _ in range(60):
            if gpu_free() < 1024:
                break
            time.sleep(10)
        r = subprocess.run(
            [PY, str(TT / 'vosr_lora_infer.py'), '-c', r'C:/models/finals/preset/ckpts/VOSR_0.5B_os',
             '--lora', r'D:/models/vosr_lora/v7/lora_s5000.pt', '--lora-scale', '0.8',
             '-i', str(VAL_IN), '-o', str(v7out), '--align_method', 'wavelet', '--seed', '8'],
            cwd=r'D:/models/vosr_lora/v7', stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f'[probe] v7 render rc={r.returncode}', flush=True)
        time.sleep(30)  # let GPU free

    # 2) eval 8 metrics per source
    for name, src, _ in SOURCES:
        if not Path(src).exists():
            print(f'[probe] {name}: src missing, skip', flush=True)
            continue
        outj = TT / f'p3a_{name}.json'
        if outj.exists():
            print(f'[probe] {name}: already done, skip', flush=True)
            continue
        print(f'[probe] eval {name} (8 metrics)...', flush=True)
        r = subprocess.run(
            [PY, EVAL, '--pred-dir', src, '--pred-pattern', '*_lq.*',
             '--gt-dir', GT_DIR, '--gt-pattern', '*_gt.*', '--out', str(outj)],
            cwd=r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace',
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f'[probe] {name} eval rc={r.returncode}', flush=True)
        if outj.exists():
            d = json.loads(outj.read_text(encoding='utf-8'))
            print(f'[probe] {name}: {json.dumps(d, ensure_ascii=False)[:400]}', flush=True)
        time.sleep(20)
    print('[done] p3a probe', flush=True)


if __name__ == '__main__':
    main()
