# -*- coding: utf-8 -*-
"""波 2 前置验证: 证明"从合并 pkg 反推 scale"这条路可用 (2026-09-10)

两个实验:
  A. 恒等复现: pkg(final_compliant_14b, sc0.93) 直接渲染 -> 应复现预登记 (0.9261, 0.3571)
     [验证 vosr_pkg_infer_14b.py 这条路 == 原来的 LoRA 路]
  B. 反推验证: remerge 到 sc0.90 -> 渲染 -> 应对上预登记 (0.9209, 0.3556)
     [验证 fp16 反推的端到端保真度]

用法: D:/Python/python.exe verify_remerge.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PY = r'D:/Python/python.exe'
TOOLS = Path(r'D:/tmp_timing')
VAL_LQ = r'D:/tmp_timing/p100_val/lq'
VAL_GT = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
BASE_C = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
OUT = Path(r'D:/tmp_timing/verify_remerge')

SRC_M0S1 = r'D:/models/final_compliant_14b/model_dir/your_model.pt'        # m0s1 @sc0.93
SRC_M0S7 = r'D:/models/final_compliant_m0s7/model_dir/your_model.pt'      # m0s7 @sc1.0

# (tag, 源 pkg, 目标 scale (None=直接用源包), 预登记 clip/lpips)
CASES = [
    ('A_identity_sc093', SRC_M0S1, None, (0.9261, 0.3571)),
    ('B_remerge_sc090', SRC_M0S1, 0.90, (0.9209, 0.3556)),
    ('C_m0s1_sc097', SRC_M0S1, 0.97, (0.9311, 0.3595)),
    ('D_m0s7_sc097', SRC_M0S7, 0.97, (0.9203, 0.3542)),
]


def run(cmd, log):
    log(f'$ {" ".join(str(c) for c in cmd)}')
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=3600)
    tail = (r.stdout or '')[-1200:] + (r.stderr or '')[-800:]
    log(tail.strip())
    return r.returncode


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG = OUT / 'verify.log'

    def log(m):
        line = f'[{time.strftime("%H:%M:%S")}] {m}'
        print(line, flush=True)
        with LOG.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    results = []
    for tag, src_pkg, rescale, expect in CASES:
        gate_early = OUT / f'gate_{tag}.json'
        if gate_early.exists():
            g = json.loads(gate_early.read_text(encoding='utf-8'))
            dc, dl = g['clip'] - expect[0], g['lpips'] - expect[1]
            log(f'CASE {tag}: 已有结果, 跳过渲染')
            results.append((tag, g['clip'], g['lpips'], expect, dc, dl))
            continue
        log('=' * 70)
        log(f'CASE {tag}  (预登记 clip/lpips = {expect})')
        pkg = src_pkg
        if rescale is not None:
            pkg = str(OUT / tag / 'model_dir' / 'your_model.pt')
            Path(pkg).parent.mkdir(parents=True, exist_ok=True)
            rc = run([PY, TOOLS / 'remerge_scale_14b.py', '--src-pkg', src_pkg,
                      '--scale', rescale, '--out', pkg], log)
            if rc != 0:
                log(f'!! remerge 失败 rc={rc}'); continue
        outdir = OUT / tag / 'render'
        rc = run([PY, TOOLS / 'vosr_pkg_infer_14b.py', '-c', BASE_C, '--pkg', pkg,
                  '--fp16', '-i', VAL_LQ, '-o', outdir, '--align_method', 'wavelet'], log)
        if rc != 0:
            log(f'!! 渲染失败 rc={rc}'); continue
        gate = OUT / f'gate_{tag}.json'
        run([PY, TOOLS / 'val_gate.py', '--src', outdir, '--val-dir', VAL_GT,
             '--clip-min', 0.919, '--lpips-max', 0.360, '--out', gate], log)
        if gate.exists():
            g = json.loads(gate.read_text(encoding='utf-8'))
            dc = g['clip'] - expect[0]
            dl = g['lpips'] - expect[1]
            log(f'==> {tag}: clip={g["clip"]:.4f} (Δ{dc:+.4f})  '
                f'lpips={g["lpips"]:.4f} (Δ{dl:+.4f})  pass={g.get("pass")}')
            results.append((tag, g['clip'], g['lpips'], expect, dc, dl))

    log('=' * 70)
    log('汇总:')
    for tag, c, l, e, dc, dl in results:
        verdict = '✅ 复现' if (abs(dc) <= 0.004 and abs(dl) <= 0.004) else '⚠️ 偏离'
        log(f'  {tag:20s} got=({c:.4f},{l:.4f}) expect={e} Δ=({dc:+.4f},{dl:+.4f}) {verdict}')
    log('VERIFY_DONE')


if __name__ == '__main__':
    main()
