# -*- coding: utf-8 -*-
"""波 0: v14 存量挖掘 (2026-09-04)
Phase 1: v14 s3200/3400/3600/3800/4000 @ sc1.0 门控 (s4000 为管线一致性校验, 期望 ~0.917/0.385)
Phase 2: 最优档 scale 扫描 {0.7, 0.8, 0.85, 0.9, 0.95} (+次优档 {0.85, 0.9})
输出: wave0_results.txt 逐行 + wave0.log
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
OUT.mkdir(exist_ok=True)
RES = OUT / 'wave0_results.txt'
LOG = OUT / 'wave0.log'

PHASE1 = [(V14 / f'lora_s{s}.pt', 1.0) for s in (4000, 3200, 3400, 3600, 3800)]
SCALES = [0.7, 0.8, 0.85, 0.9, 0.95]


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
    log(f'[boot] 波0: v14 存量挖掘, phase1={len(PHASE1)} renders')
    results = []
    for ckpt, sc in PHASE1:
        g, tag = gate_one(ckpt, sc)
        if g:
            results.append((ckpt, sc, g))
        time.sleep(2)
    # 选最优: clip>=0.9008 中 lpips 最小; 无饱和点则 clip 最高
    sat = [r for r in results if r[2]['clip'] >= 0.9008]
    pool = sat if sat else results
    best = min(pool, key=lambda r: r[2]['lpips'])
    second = sorted(pool, key=lambda r: r[2]['lpips'])[1:2]
    log(f'[phase1] 最优: {best[0].name} sc{best[1]} -> clip={best[2]["clip"]:.4f} lpips={best[2]["lpips"]:.4f}')
    # phase 2: scale 扫描
    targets = [best] + second
    for ckpt, sc0, _ in targets:
        for sc in SCALES:
            g, tag = gate_one(ckpt, sc)
            if g:
                log(f'[scale] {ckpt.name} sc{sc}: clip={g["clip"]:.4f} lpips={g["lpips"]:.4f}')
            time.sleep(2)
    log('[done] 波0 完成')


if __name__ == '__main__':
    main()
