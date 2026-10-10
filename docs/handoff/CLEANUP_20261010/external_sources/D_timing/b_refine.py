# -*- coding: utf-8 -*-
"""B lpips 精修 (2026-08-20): v2_s1500 / v3_s500 继续训练 w_clip=0 (纯 LPIPS+L1 拉保真)
每 200 步存盘 ref_s{step}.pt (800 步) -> val 门控全部 -> 唤醒
用法: D:/Python/python.exe b_refine.py
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
LOG = TT / 'b_refine.log'
OUT = Path(r'D:/models/vosr_lora')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
VAL_IN = TT / 'codiff_val_in'

JOBS = [
    ('v2', 'lora_s1500.pt', 'ref_v2'),
    ('v3', 'lora_s500.pt', 'ref_v3'),
]
STEPS = 800
SAVE_EVERY = 200


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    log('[boot] B lpips 精修启动')
    for variant, base_ck, prefix in JOBS:
        log(f'[refine] {variant}/{base_ck} -> {prefix}* (w_clip=0, {STEPS} 步, lr 1e-5)')
        with (OUT / variant / 'refine.log').open('ab') as f:
            r = subprocess.run(
                [PY, str(TT / 'train_vosr_lora.py'), '--variant', variant, '--w-clip', '0',
                 '--w-lpips', '1.0', '--w-l1', '0.05', '--steps', str(STEPS),
                 '--save-every', str(SAVE_EVERY), '--lr', '1e-5',
                 '--resume-from', str(OUT / variant / base_ck),
                 '--ckpt-prefix', prefix],
                cwd=str(TT), stdout=f, stderr=subprocess.STDOUT)
        log(f'[refine] {variant}: exit={r.returncode}')
    # val 门控全部精修 ckpt
    for variant, base_ck, prefix in JOBS:
        for ck in sorted((OUT / variant).glob(f'{prefix}_s*.pt')):
            tag = f'{variant}_{ck.stem}'
            val_out = OUT / variant / f'val_out_{tag}'
            gj = OUT / variant / f'val_gate_{tag}.json'
            log(f'[valgate] {tag}')
            subprocess.run([PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(ck),
                            '-i', str(VAL_IN), '-o', str(val_out), '--align_method', 'nofix',
                            '--seed', '42'],
                           cwd=str(OUT / variant), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(val_out), '--out', str(gj)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            gate = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else {}
            log(f'[valgate] {tag}: clip={gate.get("clip")} lpips={gate.get("lpips")} pass={gate.get("pass")}')
    log('[done] 精修门控完成, 唤醒代理')


if __name__ == '__main__':
    main()
