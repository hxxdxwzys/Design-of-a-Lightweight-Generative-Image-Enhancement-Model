# -*- coding: utf-8 -*-
"""批量验证剩余 P100 训练档位 (2026-09-02): 渲染 val 3 张 + 门控, 逐档记录
用法: D:/Python/python.exe p100_val_all.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

CKPT_DIR = Path(r'D:/tmp_timing/p100_ckpts')
OUT_DIR = Path(r'D:/tmp_timing/p100_val')
VAL_LQ = r'D:/tmp_timing/p100_val/lq'
VAL_GT = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/验证集'
INFER = r'D:/Python/python.exe D:/tmp_timing/vosr_lora_infer.py'
GATE = r'D:/Python/python.exe D:/tmp_timing/val_gate.py'
BASE_CKPT = r'C:/models/finals/preset/ckpts/VOSR_0.5B_os'
HIST = OUT_DIR / 'val_history.txt'

ITEMS = [
    ('p768_w20', 's2000'), ('p768_w20', 's2500'),
    ('p768_w15', 's1000'), ('p768_w15', 's1500'),
    ('p768_w15', 's2000'), ('p768_w15', 's2500'),
]


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with HIST.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    for v, s in ITEMS:
        ckpt = CKPT_DIR / f'{v}_lora_{s}.pt'
        tag = f'{v}_{s}'
        outdir = OUT_DIR / f'out_{tag}'
        if (OUT_DIR / f'gate_{tag}.json').exists():
            log(f'[skip] {tag} 已测')
            continue
        t0 = time.time()
        log(f'[render] {tag} ...')
        r = subprocess.run(
            f'{INFER} -c {BASE_CKPT} --lora {ckpt} -i {VAL_LQ} -o {outdir} --align_method wavelet',
            shell=True, capture_output=True, text=True, timeout=2400)
        if r.returncode != 0:
            log(f'[FAIL] {tag}: {r.stderr[-300:]}')
            continue
        g = subprocess.run(
            f'{GATE} --src {outdir} --clip-min 0.919 --lpips-max 0.360 --val-dir {VAL_GT} '
            f'--out {OUT_DIR / f"gate_{tag}.json"}',
            shell=True, capture_output=True, text=True, timeout=1200)
        try:
            res = json.loads((OUT_DIR / f'gate_{tag}.json').read_text(encoding='utf-8'))
            log(f'[GATE] {tag}: clip={res["clip"]} lpips={res["lpips"]} pass={res["pass"]} '
                f'({time.time() - t0:.0f}s)')
        except Exception as e:
            log(f'[GATE_FAIL] {tag}: {e}')
    log('[ALL_DONE]')


if __name__ == '__main__':
    main()
