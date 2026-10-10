# -*- coding: utf-8 -*-
"""VOSR 0.5B ms 多步 x cfg x v7LoRA 杂交 val 门控扫 (2026-08-24)
背景: ms 25步/cfg2.0 = (clip 0.479, lpips 0.3605) - lpips 是全场最佳 (case2=0.2062!)
      死穴是 clip。杂交思路: v7 LoRA (clip 提升器) x ms 多步 (保真器)。
组合: base x {cfg2.0, cfg0.5} + v7LoRA x {scale 1.0, 0.8}, 均 steps 8, seed 42, wavelet
结果: D:/tmp_timing/b_ms_sweep.json
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
OUT = Path(r'D:/models/vosr_ms')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_ms'
LORA = r'D:/models/vosr_lora/v7/lora_s5000.pt'
VAL_IN = TT / 'codiff_val_in'
CWD = r'C:/models/finals'  # 关键: inference_vosr 相对路径 preset/ckpts 依赖 cwd

COMBOS = [
    ('s8_cfg2.0',       dict(steps=8, cfg='2.0')),
    ('s8_cfg0.5',       dict(steps=8, cfg='0.5')),
    ('s8_cfg2.0_l1.0',  dict(steps=8, cfg='2.0', lora=LORA, scale='1.0')),
    ('s8_cfg2.0_l0.8',  dict(steps=8, cfg='2.0', lora=LORA, scale='0.8')),
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
    results = []
    for tag, kw in COMBOS:
        val_out = OUT / f'val_out_{tag}'
        gj = OUT / f'val_gate_{tag}.json'
        if gj.exists():
            results.append({'tag': tag, **json.loads(gj.read_text(encoding='utf-8'))})
            print(f'[skip] {tag} already done', flush=True)
            continue
        for _ in range(60):
            if gpu_free() < 1024:
                break
            time.sleep(10)
        t0 = time.time()
        print(f'[sweep] {tag} render val 3...', flush=True)
        args = [PY, str(TT / 'vosr_ms_infer.py'), '-c', CKPT,
                '-i', str(VAL_IN), '-o', str(val_out),
                '--upscale', '1', '--infer_steps', str(kw['steps']),
                '--cfg_scale', kw['cfg'], '--align_method', 'wavelet',
                '--tile_size', '512', '--tile_overlap', '32',
                '--vae_tile_size', '1024', '--seed', '42']
        if kw.get('lora'):
            args += ['--lora', kw['lora'], '--lora-scale', kw['scale']]
        r = subprocess.run(args, cwd=CWD, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f'[sweep] {tag} render FAILED rc={r.returncode}', flush=True)
            results.append({'tag': tag, 'ok': False})
            continue
        subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(val_out), '--out', str(gj)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        gate = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else {}
        results.append({'tag': tag, **gate})
        print(f'[sweep] {tag}: clip={gate.get("clip")} lpips={gate.get("lpips")} ({time.time() - t0:.0f}s)', flush=True)
    (TT / 'b_ms_sweep.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('[done] b_ms_sweep.json', flush=True)


if __name__ == '__main__':
    main()
