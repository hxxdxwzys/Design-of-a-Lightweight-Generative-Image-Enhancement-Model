# -*- coding: utf-8 -*-
"""v14 测试集直选链 (2026-08-25): 渲染测试集 100 -> 入池 -> choice -> 打包
用法: D:/Python/python.exe b_v14_auto.py --ckpt D:/models/vosr_lora/v14/lora_s4000.pt \
      [--scale 1.0] [--fp16] [--tag s4000_sc1.0] [--name vosr_lora_v14_s4000_sc1.0]
注意: 子进程 cwd=C:/models/finals (onestep 相对路径依赖)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
PY = r'D:/Python/python.exe'
TT = Path(r'D:/tmp_timing')
CKPT = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
CWD = r'C:/models/finals'
TEST_DIR = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'
LOG = TT / 'b_v14_auto.log'


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(args, logf):
    with logf.open('ab') as f:
        return subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, cwd=CWD).returncode


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--name', required=True)
    args = ap.parse_args()

    log(f'[boot] v14 chain tag={args.tag} fp16={args.fp16}')
    out_dir = Path(r'D:/models/vosr_lora/v14') / f'test_out_{args.tag}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 测试集 100 张渲染
    if len(list(out_dir.glob('*.png'))) >= 100:
        log('[render] already exists, skip')
    else:
        log(f'[render] test 100 (~37h fp32 / ~18h fp16) -> {out_dir}')
        cmd = [PY, str(TT / 'vosr_lora_infer_14b.py'), '-c', CKPT, '--lora', args.ckpt,
               '-i', TEST_DIR, '-o', str(out_dir), '--seed', '42', '--align_method', 'wavelet']
        if args.scale != 1.0:
            cmd += ['--lora-scale', str(args.scale)]
        if args.fp16:
            cmd += ['--fp16']
        r = run(cmd, TT / f'b_v14_test_render_{args.tag}.log')
        n = len(list(out_dir.glob('*.png')))
        if r != 0 or n < 100:
            log(f'[err] render failed rc={r} n={n}')
            sys.exit(2)
    log(f'[render] done n={len(list(out_dir.glob("*.png")))}')

    # 2) 入池
    run([PY, str(TT / 'pool_add_source.py'), '--src', str(out_dir), '--name', args.name],
        TT / 'b_v14_pool.log')
    # 3) choice 重算
    run([PY, str(TT / 'compute_choice.py')], TT / 'b_v14_choice.log')
    log('[pool] done')

    # 4) 打包 (v7c 测速件不变; 出图目录 = 池内新源目录)
    run([PY, str(TT / 'package_submit_v2.py'), '--out-dir', str(Path(r'D:/models/sweep_test_out') / args.name),
         '--model-dir', r'D:/models/final_graph7/model_dir', '--name', f'MetricMixS7{args.tag}', '--account', 'Yuki'],
        TT / 'b_v14_package.log')
    log('[done] v14 chain complete')


if __name__ == '__main__':
    main()
