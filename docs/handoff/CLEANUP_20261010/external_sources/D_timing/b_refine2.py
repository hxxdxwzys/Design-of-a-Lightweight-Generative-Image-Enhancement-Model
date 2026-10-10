# -*- coding: utf-8 -*-
"""B 接力 2 (2026-08-20): 等 b_refine 完成 -> 从基础 VOSR 纯保真训练 v4 (w_clip=0, 2500 步)
-> val 门控 v4 ckpts -> 汇总唤醒
用法: D:/Python/python.exe b_refine2.py
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
LOG = TT / 'b_refine2.log'
OUT = Path(r'D:/models/vosr_lora')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
VAL_IN = TT / 'codiff_val_in'


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
    log('[boot] 等待 b_refine 完成...')
    t0 = time.time()
    refine_log = TT / 'b_refine.log'
    while time.time() - t0 < 8 * 3600:
        if refine_log.exists() and '[done] 精修门控完成' in refine_log.read_text(encoding='utf-8', errors='ignore'):
            break
        time.sleep(60)
    log('[step1] b_refine 完成, 开始 v4 纯保真训练 (w_clip=0, 2500 步, 每 500 存盘)')
    with (OUT / 'v4' / 'train.log').open('ab') as f:
        r = subprocess.run(
            [PY, str(TT / 'train_vosr_lora.py'), '--variant', 'v4', '--w-clip', '0',
             '--w-lpips', '1.0', '--w-l1', '0.05', '--steps', '2500',
             '--save-every', '500', '--lr', '3e-5', '--ckpt-prefix', 'v4'],
            cwd=str(TT), stdout=f, stderr=subprocess.STDOUT)
    log(f'[step1] v4 训练 exit={r.returncode}')
    for ck in sorted((OUT / 'v4').glob('v4_s*.pt')):
        tag = f'v4_{ck.stem}'
        val_out = OUT / 'v4' / f'val_out_{tag}'
        gj = OUT / 'v4' / f'val_gate_{tag}.json'
        log(f'[valgate] {tag}')
        subprocess.run([PY, str(TT / 'vosr_lora_infer.py'), '-c', CKPT, '--lora', str(ck),
                        '-i', str(VAL_IN), '-o', str(val_out), '--align_method', 'nofix',
                        '--seed', '42'],
                       cwd=str(OUT / 'v4'), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.run([PY, str(TT / 'val_gate.py'), '--src', str(val_out), '--out', str(gj)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        gate = json.loads(gj.read_text(encoding='utf-8')) if gj.exists() else {}
        log(f'[valgate] {tag}: clip={gate.get("clip")} lpips={gate.get("lpips")} pass={gate.get("pass")}')
    log('[done] v4 门控完成, 唤醒代理')


if __name__ == '__main__':
    main()
