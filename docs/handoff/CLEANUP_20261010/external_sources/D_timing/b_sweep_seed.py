# -*- coding: utf-8 -*-
"""sc0.8 多种子 val 扫参 (2026-08-21): seed {1..5} x v5_s4000 scale0.8 val wavelet 门控
目的: 找 lpips < 0.3867 (M20) 且 clip >= 0.902 的 seed -> M22 候选
结果: D:/tmp_timing/b_sweep_seed.json
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
PY = r'D:/Python/python.exe'
TT = Path(r'D:/tmp_timing')
OUT = Path(r'D:/models/vosr_lora/v5')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
VAL_IN = TT / 'codiff_val_in'

SEEDS = [1, 2, 3, 4, 5, 7]


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    results = []
    for seed in SEEDS:
        tag = f'sc08_seed{seed}'
        val_out = OUT / f'val_out_{tag}_wavelet'
        gj = OUT / f'val_gate_{tag}_wavelet.json'
        t0 = time.time()
        print(f'[sweep] {tag} 渲染 val 3...', flush=True)
        r = subprocess.run(
            [PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(OUT / 'v5_s4000.pt'),
             '--lora-scale', '0.8', '-i', str(VAL_IN), '-o', str(val_out),
             '--align_method', 'wavelet', '--seed', str(seed)],
            cwd=str(OUT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f'[sweep] {tag} 渲染失败', flush=True)
            results.append({'tag': tag, 'ok': False})
            continue
        subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(val_out), '--out', str(gj)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        gate = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else {}
        results.append({'tag': tag, 'seed': seed, 'ok': True, **gate})
        print(f'[sweep] {tag}: clip={gate.get("clip")} lpips={gate.get("lpips")} ({time.time() - t0:.0f}s)', flush=True)
    (TT / 'b_sweep_seed.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('[done] b_sweep_seed.json', flush=True)


if __name__ == '__main__':
    main()
