# -*- coding: utf-8 -*-
"""B LoRA 强度插值扫参 (2026-08-20): 找 clipiqa×lpips 前沿甜点
W' = W + scale * LoRA; 对 (v2_s1500, v3_s500) × scale(0.3/0.5/0.7) 渲染 val 3 + 门控
输出: D:/tmp_timing/b_sweep.json
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
OUT = Path(r'D:/models/vosr_lora')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
VAL_IN = TT / 'codiff_val_in'

COMBOS = [
    ('v2', 'lora_s1500.pt', 0.3),
    ('v2', 'lora_s1500.pt', 0.5),
    ('v2', 'lora_s1500.pt', 0.7),
    ('v3', 'lora_s500.pt', 0.3),
    ('v3', 'lora_s500.pt', 0.5),
    ('v3', 'lora_s500.pt', 0.7),
]


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    results = []
    for variant, ck, scale in COMBOS:
        tag = f'{variant}_{ck[:-3]}_sc{scale}'
        val_out = OUT / variant / f'val_out_{tag}'
        gj = OUT / variant / f'val_gate_{tag}.json'
        t0 = time.time()
        print(f'[sweep] {tag} 渲染 val 3...', flush=True)
        r = subprocess.run(
            [PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(OUT / variant / ck),
             '--lora-scale', str(scale), '-i', str(VAL_IN), '-o', str(val_out),
             '--align_method', 'nofix', '--seed', '42'],
            cwd=str(OUT / variant), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f'[sweep] {tag} 渲染失败', flush=True)
            results.append({'tag': tag, 'ok': False})
            continue
        subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(val_out), '--out', str(gj)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        gate = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else {}
        results.append({'tag': tag, 'variant': variant, 'ck': ck, 'scale': scale,
                        'ok': True, **gate})
        print(f'[sweep] {tag}: clip={gate.get("clip")} lpips={gate.get("lpips")} ({time.time() - t0:.0f}s)', flush=True)
    (TT / 'b_sweep.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('[done] b_sweep.json', flush=True)
    print(json.dumps(results, indent=1), flush=True)


if __name__ == '__main__':
    main()
