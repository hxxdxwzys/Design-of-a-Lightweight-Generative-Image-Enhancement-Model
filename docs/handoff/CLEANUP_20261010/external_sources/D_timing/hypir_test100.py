# -*- coding: utf-8 -*-
"""Stage 1: 用 HYPIR 跑 100 张测试集 (2026-09-13) —— 不花提交次数
目的: 判断 HYPIR 的"干净复原"优势在测试集上是否成立, 再决定要不要投入 1-2 天做合规集成。
参数来自旧脚本 workspace/tmp_e4_gate.py 的已验证调用 (model_t/coeff_t=200, lora_rank=256, captioner=empty)。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = r'D:\Python\python.exe'
BASE = r'D:/models/hypir_code/HYPIR-main/HYPIR-main'
LQ = Path(r'D:/models/hypir/lq_test100')
OUT = Path(r'D:/models/hypir/out_test100')
TEST = Path(r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集')
WT = r'D:/models/hypir/HYPIR_sd2.pth'
LOG = Path(r'D:/tmp_timing/hypir_test100.log')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def main():
    if not LQ.exists() or len(list(LQ.glob('*'))) < 100:
        LQ.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(TEST.glob('*.jpg')):
            d = LQ / f'{f.stem}.png'
            if not d.exists():
                from PIL import Image
                Image.open(f).convert('RGB').save(d)
            n += 1
        print(f'[prep] {n} 张 -> {LQ}', flush=True)
    print(f'[prep] 输入 {len(list(LQ.glob("*")))} 张', flush=True)

    cmd = [PY, 'test.py',
           '--base_model_type', 'sd2', '--base_model_path', r'D:/models/sd21base',
           '--model_t', '200', '--coeff_t', '200',
           '--lora_rank', '256',
           '--lora_modules', 'to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,'
                             'conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj',
           '--weight_path', WT,
           '--patch_size', '512', '--stride', '256',
           '--lq_dir', str(LQ), '--upscale', '1', '--captioner', 'empty',
           '--output_dir', str(OUT), '--device', 'cuda']
    print('>>', ' '.join(cmd), flush=True)
    t0 = time.time()
    with LOG.open('w', encoding='utf-8') as fo:
        r = subprocess.run(cmd, cwd=BASE,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        fo.write(r.stdout or '')
    print(f'[done] rc={r.returncode}, {(time.time()-t0)/60:.1f} min', flush=True)
    res = OUT / 'result'
    n = len(list(res.glob('*'))) if res.exists() else 0
    print(f'[out] {n} 张 -> {res}', flush=True)
    if n < 100:
        print('[tail] ' + '\n'.join((r.stdout or '').splitlines()[-15:]), flush=True)
    print('HYPIR_TEST100_DONE', flush=True)


if __name__ == '__main__':
    main()
