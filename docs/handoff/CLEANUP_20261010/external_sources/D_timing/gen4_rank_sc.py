# -*- coding: utf-8 -*-
"""rank 轴 scale 微扫 (2026-09-09 18:00, 宽容协议第 2 步)
r16 最优档 s2000 / 最高 clip 档 s4000 × sc {0.90, 0.93, 0.95} -> 找各自 (clip>=0.920, lpips min)
先下载 ckpt (autoval 门控后已删本地缓存)。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, r'D:/tmp_timing')

BASE = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
VAL_LQ = r'D:/tmp_timing/p100_val/lq'
VAL_GT = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
CKPT_DIR = Path(r'D:/tmp_timing/p100_ckpts')
OUT = Path(r'D:/tmp_timing/rank_sc')
OUT.mkdir(exist_ok=True)
JOBS = [('p512_14b_m0s1r16', 2000), ('p512_14b_m0s1r16', 4000)]
SCALES = [0.90, 0.93, 0.95]
LOG = OUT / 'gen4_rank_sc.log'
RES = OUT / 'gen4_rank_sc_results.txt'


def log(m):
    line = time.strftime('[%m-%d %H:%M:%S] ') + m
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def download(variant, step):
    local = CKPT_DIR / f'{variant}_lora_s{step}.pt'
    if local.exists():
        log(f'[已有] {local.name}')
        return local
    from p100_upload import connect
    log(f'[下载] {variant}/lora_s{step}.pt')
    cli = connect()
    sftp = cli.open_sftp()
    sftp.get(f'p100/runs/{variant}/lora_s{step}.pt', str(local))
    sftp.close()
    cli.close()
    log(f'[下载OK] {local.stat().st_size} bytes')
    return local


def main():
    log('[boot] rank 轴 scale 微扫 (r16 s2000/s4000)')
    for variant, step in JOBS:
        ckpt = download(variant, step)
        for sc in SCALES:
            tag = f'{variant}_s{step}_sc{sc}'
            outdir = OUT / f'out_{tag}'
            gate = OUT / f'gate_{tag}.json'
            if gate.exists():
                g = json.loads(gate.read_text(encoding='utf-8'))
                log(f'[已有] {tag}: clip={g["clip"]:.4f} lpips={g["lpips"]:.4f}')
                continue
            log(f'[渲染] {tag}')
            r1 = subprocess.run(
                f'D:/Python/python.exe D:/tmp_timing/vosr_lora_infer_14b.py -c {BASE} '
                f'--lora {ckpt} --lora-scale {sc} --fp16 -i {VAL_LQ} -o {outdir} '
                f'--align_method wavelet',
                shell=True, capture_output=True, text=True, timeout=10800)
            if r1.returncode != 0:
                log(f'[渲染失败] {tag}: {r1.stderr[-300:]}')
                continue
            subprocess.run(
                f'D:/Python/python.exe D:/tmp_timing/val_gate.py --src {outdir} '
                f'--clip-min 0.919 --lpips-max 0.360 --val-dir {VAL_GT} --out {gate}',
                shell=True, capture_output=True, text=True, timeout=1800)
            try:
                g = json.loads(gate.read_text(encoding='utf-8'))
                log(f'[门控] {tag}: clip={g["clip"]:.4f} lpips={g["lpips"]:.4f} pass={g["pass"]}')
                with RES.open('a', encoding='utf-8') as f:
                    f.write(f'{tag}\t{g["clip"]:.4f}\t{g["lpips"]:.4f}\t{g["pass"]}\n')
            except Exception as e:
                log(f'[门控异常] {tag}: {e}')
    log('[done] rank 轴 scale 微扫完成')


if __name__ == '__main__':
    main()
