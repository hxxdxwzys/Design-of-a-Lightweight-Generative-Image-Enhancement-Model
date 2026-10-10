# -*- coding: utf-8 -*-
"""世代3 完整评估 (2026-09-07): scale 微扫 + 宽容期检测
对世代 3 各线最优档做 scale 扫描, 找各自 (clip>=0.920, lpips min);
之后由调用方对最优档做 8 指标健康检查。
"""
import json
import subprocess
import time
from pathlib import Path

BASE = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
VAL_LQ = r'D:/tmp_timing/p100_val/lq'
VAL_GT = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
OUT = Path(r'D:/tmp_timing/wave0')
RES = OUT / 'gen3_scale_results.txt'
LOG = OUT / 'gen3_scale.log'

# (ckpt, 名字, scales)
JOBS = [
    (Path(r'D:/tmp_timing/p100_ckpts/p512_14b_g3w18s1_lora_s4500.pt'), 'g3w18s1_s4500', [1.0, 0.97, 0.95, 0.93]),
    (Path(r'D:/tmp_timing/p100_ckpts/p512_14b_g3w20s7_lora_s6500.pt'), 'g3w20s7_s6500', [1.0, 0.97, 0.95, 0.93]),
]


def log(m):
    line = time.strftime('[%m-%d %H:%M:%S] ') + m
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def gate_one(ckpt, name, sc):
    tag = f'{name}_sc{sc}'
    outdir = OUT / f'out_{tag}'
    gate = OUT / f'gate_{tag}.json'
    if gate.exists():
        g = json.loads(gate.read_text(encoding='utf-8'))
        log(f'[已有] {tag}: clip={g["clip"]:.4f} lpips={g["lpips"]:.4f}')
        return g
    log(f'[渲染] {tag}')
    r1 = subprocess.run(
        f'D:/Python/python.exe D:/tmp_timing/vosr_lora_infer_14b.py -c {BASE} '
        f'--lora {ckpt} --lora-scale {sc} --fp16 -i {VAL_LQ} -o {outdir} --align_method wavelet',
        shell=True, capture_output=True, text=True, timeout=10800)
    if r1.returncode != 0:
        log(f'[渲染失败] {tag}: {r1.stderr[-200:]}')
        return None
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
        return g
    except Exception as e:
        log(f'[门控异常] {tag}: {e}')
        return None


def main():
    log('[boot] 世代3 scale 微扫 (宽容期评估)')
    for ckpt, name, scales in JOBS:
        log(f'=== {name} ===')
        for sc in scales:
            gate_one(ckpt, name, sc)
    log('[done] 世代3 scale 微扫完成')


if __name__ == '__main__':
    main()
