# -*- coding: utf-8 -*-
"""世代2: m0s1-s4000 scale 微扫 (2026-09-06): sc {0.9, 0.93, 0.95, 0.97}
目标: clip>=0.920 下 lpips<=0.360 (sc0.95 预估 (0.921, 0.359))
"""
import json
import subprocess
import time
from pathlib import Path

BASE = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
VAL_LQ = r'D:/tmp_timing/p100_val/lq'
VAL_GT = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
CKPT = Path(r'D:/tmp_timing/p100_ckpts/p512_14b_m0s1_lora_s4000.pt')
OUT = Path(r'D:/tmp_timing/wave0')
SCALES = [0.9, 0.93, 0.95, 0.97]
RES = OUT / 'gen2_sc_results.txt'
LOG = OUT / 'gen2_sc.log'


def log(m):
    line = time.strftime('[%m-%d %H:%M:%S] ') + m
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    log('[boot] gen2 sc 微扫 m0s1-s4000')
    for sc in SCALES:
        tag = f'm0s1_s4000_sc{sc}'
        outdir = OUT / f'out_{tag}'
        gate = OUT / f'gate_{tag}.json'
        if gate.exists():
            g = json.loads(gate.read_text(encoding='utf-8'))
            log(f'[已有] {tag}: clip={g["clip"]:.4f} lpips={g["lpips"]:.4f}')
            continue
        log(f'[渲染] {tag}')
        r1 = subprocess.run(
            f'D:/Python/python.exe D:/tmp_timing/vosr_lora_infer_14b.py -c {BASE} '
            f'--lora {CKPT} --lora-scale {sc} --fp16 -i {VAL_LQ} -o {outdir} --align_method wavelet',
            shell=True, capture_output=True, text=True, timeout=10800)
        if r1.returncode != 0:
            log(f'[渲染失败] {tag}: {r1.stderr[-200:]}')
            continue
        subprocess.run(
            f'D:/Python/python.exe D:/tmp_timing/val_gate.py --src {outdir} '
            f'--clip-min 0.919 --lpips-max 0.360 --val-dir {VAL_GT} --out {gate}',
            shell=True, capture_output=True, text=True, timeout=1800)
        try:
            g = json.loads(gate.read_text(encoding='utf-8'))
            line = f'{tag}: clip={g["clip"]:.4f} lpips={g["lpips"]:.4f}'
            log(f'[门控] {line}')
            with RES.open('a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception as e:
            log(f'[门控异常] {tag}: {e}')
    log('[done] gen2 sc 微扫完成')


if __name__ == '__main__':
    main()
