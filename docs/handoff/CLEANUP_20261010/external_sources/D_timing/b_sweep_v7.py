# -*- coding: utf-8 -*-
"""v7 val 门控 + scale 档 (2026-08-22): v7 6 ckpt x scale {0.8,0.9,1.0} val wavelet
目的: 找 (clip>=0.90, lpips 最低) 的 M23 候选, 按新模型打分 (饱和区 lpips 主轴)
结果: D:/tmp_timing/b_sweep_v7.json
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

COMBOS = [(s, sc) for s in (1000, 2000, 3000, 4000, 5000, 6000) for sc in (0.8, 0.9, 1.0)]


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
    for step, scale in COMBOS:
        tag = f'v7_s{step}_sc{scale}'
        val_out = OUT / f'val_out_{tag}_wavelet'
        gj = OUT / f'val_gate_{tag}_wavelet.json'
        if gj.exists():
            results.append({'tag': tag, 'step': step, 'scale': scale, **json.loads(gj.read_text(encoding='utf-8'))})
            print(f'[skip] {tag} 已测', flush=True)
            continue
        for _ in range(60):
            if gpu_free() < 1024:
                break
            time.sleep(10)
        t0 = time.time()
        print(f'[sweep] {tag} 渲染 val 3...', flush=True)
        args = [PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(OUT / f'lora_s{step}.pt'),
                '-i', str(VAL_IN), '-o', str(val_out), '--align_method', 'wavelet', '--seed', '42']
        if scale != 1.0:
            args += ['--lora-scale', str(scale)]
        r = subprocess.run(args, cwd=str(OUT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f'[sweep] {tag} 渲染失败 rc={r.returncode}', flush=True)
            results.append({'tag': tag, 'step': step, 'scale': scale, 'ok': False})
            continue
        subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(val_out), '--out', str(gj)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        gate = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else {}
        results.append({'tag': tag, 'step': step, 'scale': scale, **gate})
        print(f'[sweep] {tag}: clip={gate.get("clip")} lpips={gate.get("lpips")} ({time.time() - t0:.0f}s)', flush=True)
    (TT / 'b_sweep_v7.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('[done] b_sweep_v7.json', flush=True)


if __name__ == '__main__':
    main()
