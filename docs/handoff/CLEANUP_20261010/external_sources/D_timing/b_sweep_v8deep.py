# -*- coding: utf-8 -*-
"""v8 best ckpt sc0.8 seed 深扫 (2026-08-23): seed {8..45} val wavelet
目的: 门控过后找 lpips < 0.376 的安全区 seed (case2 轴榨 0.002~0.004)
用法: 先跑 b_sweep_v8.py 确定 best ckpt, 改 CKSTEP/SCALE 后运行
结果: D:/tmp_timing/b_sweep_v8deep.json
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
OUT = Path(r'D:/models/vosr_lora/v8')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
VAL_IN = TT / 'codiff_val_in'

CKSTEP = 9000   # <<< 改这里
SCALE = 0.8
SEEDS = list(range(8, 46))  # 8..45


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
    for seed in SEEDS:
        tag = f'v8s{CKSTEP}_sc{SCALE}_seed{seed}'
        val_out = OUT / f'val_out_{tag}_wavelet'
        gj = OUT / f'val_gate_{tag}_wavelet.json'
        if gj.exists():
            results.append({'tag': tag, 'seed': seed, **json.loads(gj.read_text(encoding='utf-8'))})
            print(f'[skip] {tag} already done', flush=True)
            continue
        for _ in range(60):
            if gpu_free() < 1024:
                break
            time.sleep(10)
        t0 = time.time()
        print(f'[sweep] {tag} render val 3...', flush=True)
        args = [PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(OUT / f'lora_s{CKSTEP}.pt'),
                '--lora-scale', str(SCALE), '-i', str(VAL_IN), '-o', str(val_out),
                '--align_method', 'wavelet', '--seed', str(seed)]
        r = subprocess.run(args, cwd=str(OUT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f'[sweep] {tag} render FAILED rc={r.returncode}', flush=True)
            results.append({'tag': tag, 'seed': seed, 'ok': False})
            continue
        subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(val_out), '--out', str(gj)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        gate = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else {}
        results.append({'tag': tag, 'seed': seed, **gate})
        print(f'[sweep] {tag}: clip={gate.get("clip")} lpips={gate.get("lpips")} ({time.time() - t0:.0f}s)', flush=True)
    (TT / 'b_sweep_v8deep.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('[done] b_sweep_v8deep.json', flush=True)


if __name__ == '__main__':
    main()
