# -*- coding: utf-8 -*-
"""波 0b: v14 细扫 s2400-s3000 (每200步) sc1.0, 找比 s3200 (0.3786) 更低的点
命中则对最优档做 sc0.95 复扫。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
VAL_LQ = r'D:/tmp_timing/p100_val/lq'
VAL_GT = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
V14 = Path(r'D:/models/vosr_lora/v14')
OUT = Path(r'D:/tmp_timing/wave0')
RES = OUT / 'wave0b_results.txt'
LOG = OUT / 'wave0b.log'
STEPS = [2400, 2600, 2800, 3000]


def log(m):
    line = time.strftime('[%m-%d %H:%M:%S] ') + m
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def gate_one(ckpt, scale):
    tag = f'{ckpt.parent.name}_s{ckpt.stem.split("_")[-1]}_sc{scale}'
    outdir = OUT / f'out_{tag}'
    if (OUT / f'gate_{tag}.json').exists():
        return json.loads((OUT / f'gate_{tag}.json').read_text(encoding='utf-8')), tag
    log(f'[渲染] {tag} (scale={scale})')
    r1 = subprocess.run(
        f'D:/Python/python.exe D:/tmp_timing/vosr_lora_infer_14b.py -c {BASE} '
        f'--lora {ckpt} --lora-scale {scale} --fp16 -i {VAL_LQ} -o {outdir} --align_method wavelet',
        shell=True, capture_output=True, text=True, timeout=10800)
    if r1.returncode != 0:
        log(f'[渲染失败] {tag}: {r1.stderr[-300:]}')
        return None, tag
    gate = OUT / f'gate_{tag}.json'
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
        return g, tag
    except Exception as e:
        log(f'[门控异常] {tag}: {e}')
        return None, tag


def main():
    log(f'[boot] 波0b: v14 细扫 {STEPS}')
    best = None
    for s in STEPS:
        ckpt = V14 / f'lora_s{s}.pt'
        g, _ = gate_one(ckpt, 1.0)
        if g and (best is None or g['lpips'] < best[1]['lpips']):
            best = (ckpt, g)
    if best:
        ckpt, g = best
        log(f'[最优] {ckpt.name} sc1.0: clip={g["clip"]:.4f} lpips={g["lpips"]:.4f}')
        for sc in (0.95,):
            g2, _ = gate_one(ckpt, sc)
            if g2:
                log(f'[scale] {ckpt.name} sc{sc}: clip={g2["clip"]:.4f} lpips={g2["lpips"]:.4f}')
    log('[done] 波0b 完成')


if __name__ == '__main__':
    main()
