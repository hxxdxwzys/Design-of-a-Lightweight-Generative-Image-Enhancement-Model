# -*- coding: utf-8 -*-
"""FiDeSR val 驱动 (2026-08-25): val 3 张渲染 + 门控
用法: cd FiDeSR-main && D:/Python/python.exe D:/tmp_timing/b_fidesr_val.py [--upscale 1|4] [--lf 0.2] [--hf 0.3] [--tag default]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
TT = Path(r'D:/tmp_timing')
FD = r'D:/models/new_src/fidesr/code/FiDeSR-main'
SD = r'C:/models/finals/preset/ckpts/stable-diffusion-2-1-base'
PKL = r'D:/models/new_src/fidesr/fidesr.pkl'
VAL_IN = r'D:/tmp_timing/codiff_val_in'
OUT_ROOT = Path(r'D:/models/new_src/fidesr')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--upscale', type=int, default=1)
    ap.add_argument('--lf', type=float, default=0.2)
    ap.add_argument('--hf', type=float, default=0.3)
    ap.add_argument('--tag', default='v1')
    args = ap.parse_args()

    out = OUT_ROOT / f'val_out_u{args.upscale}_lf{args.lf}_hf{args.hf}_{args.tag}'
    gj = OUT_ROOT / f'val_gate_u{args.upscale}_lf{args.lf}_hf{args.hf}_{args.tag}.json'
    if gj.exists():
        print(f'[skip] {gj.stem} already done', flush=True)
        return
    cmd = [PY, 'test_fidesr.py',
           '--pretrained_model_path', SD, '--pretrained_path', PKL,
           '--input_image', VAL_IN, '--output_dir', str(out),
           '--upscale', str(args.upscale), '--align_method', 'wavelet',
           '--lf_scale', str(args.lf), '--hf_scale', str(args.hf),
           '--mixed_precision', 'fp16', '--seed', '42']
    t0 = time.time()
    r = subprocess.run(cmd, cwd=FD)
    print(f'[render] rc={r.returncode} ({time.time()-t0:.0f}s)', flush=True)
    if r.returncode != 0:
        print('[FAIL] render failed', flush=True)
        sys.exit(2)
    subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(out), '--out', str(gj)])
    if gj.exists():
        d = json.loads(gj.read_text(encoding='utf-8'))
        print(f'[gate] clip={d["clip"]} lpips={d["lpips"]}', flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
