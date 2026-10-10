# -*- coding: utf-8 -*-
"""重跑失败 seed (2026-08-21): seed {2,3,4,5,7} x v5_s4000 scale0.8 val wavelet
修复: 每 seed 前等 GPU 显存 <1GB (前序进程 CUDA 上下文释放)
结果: 合并进 D:/tmp_timing/b_sweep_seed.json
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

SEEDS = [2, 3, 4, 5, 7]


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
    all_res = json.loads((TT / 'b_sweep_seed.json').read_text(encoding='utf-8'))
    have = {r.get('seed') for r in all_res if r.get('ok')}
    for seed in SEEDS:
        if seed in have:
            print(f'[skip] seed{seed} 已有结果', flush=True)
            continue
        tag = f'sc08_seed{seed}'
        val_out = OUT / f'val_out_{tag}_wavelet'
        gj = OUT / f'val_gate_{tag}_wavelet.json'
        t0 = time.time()
        # 等 GPU 空闲 (前序进程释放)
        for _ in range(60):
            if gpu_free() < 1024:
                break
            time.sleep(10)
        print(f'[sweep] {tag} 渲染 val 3...', flush=True)
        r = subprocess.run(
            [PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(OUT / 'v5_s4000.pt'),
             '--lora-scale', '0.8', '-i', str(VAL_IN), '-o', str(val_out),
             '--align_method', 'wavelet', '--seed', str(seed)],
            cwd=str(OUT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f'[sweep] {tag} 渲染失败 rc={r.returncode}', flush=True)
            all_res.append({'tag': tag, 'seed': seed, 'ok': False})
            continue
        subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(val_out), '--out', str(gj)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        gate = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else {}
        all_res.append({'tag': tag, 'seed': seed, 'ok': True, **gate})
        print(f'[sweep] {tag}: clip={gate.get("clip")} lpips={gate.get("lpips")} ({time.time() - t0:.0f}s)', flush=True)
    (TT / 'b_sweep_seed.json').write_text(json.dumps(all_res, indent=1), encoding='utf-8')
    print('[done] b_sweep_seed.json merged', flush=True)


if __name__ == '__main__':
    main()
