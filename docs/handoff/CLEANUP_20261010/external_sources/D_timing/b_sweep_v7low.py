# -*- coding: utf-8 -*-
"""v7_s5000 低 scale 前沿扫 (2026-08-22): scale {0.6,0.7,0.75} x seed {1,6} val wavelet
目的: 找 (clip>=0.90, lpips<=0.38) 的 M24 候选 (lpips 低段权重 -1150/unit 更陡)
结果: D:/tmp_timing/b_sweep_v7low.json
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
OUT = Path(r'D:/models/vosr_lora/v7')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
VAL_IN = TT / 'codiff_val_in'

COMBOS = [(sc, sd) for sc in (0.6, 0.7, 0.75) for sd in (1, 6)]


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
    results = []
    for scale, seed in COMBOS:
        tag = f'v7s5000_sc{scale}_seed{seed}'
        val_out = OUT / f'val_out_{tag}_wavelet'
        gj = OUT / f'val_gate_{tag}_wavelet.json'
        if gj.exists():
            results.append({'tag': tag, 'scale': scale, 'seed': seed, **json.loads(gj.read_text(encoding='utf-8'))})
            print(f'[skip] {tag} 已测', flush=True)
            continue
        for _ in range(60):
            if gpu_free() < 1024:
                break
            time.sleep(10)
        t0 = time.time()
        print(f'[sweep] {tag} 渲染 val 3...', flush=True)
        r = subprocess.run(
            [PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(OUT / 'lora_s5000.pt'),
             '--lora-scale', str(scale), '-i', str(VAL_IN), '-o', str(val_out),
             '--align_method', 'wavelet', '--seed', str(seed)],
            cwd=str(OUT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f'[sweep] {tag} 渲染失败 rc={r.returncode}', flush=True)
            results.append({'tag': tag, 'scale': scale, 'seed': seed, 'ok': False})
            continue
        subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(val_out), '--out', str(gj)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        gate = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else {}
        results.append({'tag': tag, 'scale': scale, 'seed': seed, **gate})
        print(f'[sweep] {tag}: clip={gate.get("clip")} lpips={gate.get("lpips")} ({time.time() - t0:.0f}s)', flush=True)
    (TT / 'b_sweep_v7low.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('[done] b_sweep_v7low.json', flush=True)


if __name__ == '__main__':
    main()
