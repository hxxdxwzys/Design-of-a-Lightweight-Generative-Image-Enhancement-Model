# -*- coding: utf-8 -*-
"""VOSR 0.5B os 多种子测试集渲染 (通宵): 5 个新种子, 输出 preset/vosr_seed{seed}_test/
之后: 转 jpg 入池 (vosr_seeds_to_pool.py) + 逐图指标 + compute_choice 重算 -> M16
用法: D:/Python/python.exe seed_runner.py
"""
import os
import subprocess
import time

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
CWD = r'C:/models/finals'
SCRIPT = r'D:/models/finals/VOSR/inference_vosr_onestep.py'
CKPT = r'preset/ckpts/VOSR_0.5B_os'
TEST = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'
SEEDS = [111, 222, 333, 444, 555]


def main():
    for seed in SEEDS:
        out = f'preset/vosr_seed{seed}_test'
        cmd = [r'D:/Python/python.exe', SCRIPT, '-c', CKPT, '-i', TEST, '-o', out,
               '-u', '1', '--tile_size', '512', '--vae_tile_size', '1024',
               '--vae_tile_overlap', '64', '--seed', str(seed)]
        print(f'[seed {seed}] start {time.strftime("%H:%M:%S")}', flush=True)
        t0 = time.time()
        subprocess.run(cmd, cwd=CWD, check=False)
        print(f'[seed {seed}] done {time.strftime("%H:%M:%S")} ({time.time() - t0:.0f}s)', flush=True)
    print('[all seeds done]', flush=True)


if __name__ == '__main__':
    main()
