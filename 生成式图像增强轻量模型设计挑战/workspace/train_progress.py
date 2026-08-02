#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地 4060 训练实时进度条。

数据源:
  1. 若 output_dir 下有 train_progress.txt (train_local.py --progress_file 写入) → 平滑逐条
  2. 否则从最新 checkpoint unet_step{N}.pth 文件名估 step (粗粒度, 每1000步)
  3. nvidia-smi 显示 GPU 利用/显存/温度

用法:
  python train_progress.py                          # 动画 (每2秒刷新, Ctrl+C 退出)
  python train_progress.py once                     # 一次性快照
  python train_progress.py --dir <dir> [--iters N]  # 指定目录/总步数
"""
import os
import re
import glob
import time
import argparse
import subprocess


def checkpoint_steps(out_dir):
    """返回已保存 checkpoint 的 step 列表。"""
    steps = []
    for f in glob.glob(os.path.join(out_dir, 'unet_step*.pth')):
        m = re.search(r'unet_step(\d+)\.pth$', f)
        if m:
            steps.append((int(m.group(1)), os.path.getmtime(f)))
    return sorted(steps)


def read_progress_file(out_dir):
    """读 train_progress.txt (每行: step loss... 与训练 stdout 同格式)。"""
    p = os.path.join(out_dir, 'train_progress.txt')
    if not os.path.exists(p):
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return f.read().splitlines()


def gpu_info():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10)
        util, mem, temp = r.stdout.strip().split(',')
        return f'GPU {util.strip():>3}% | {mem.strip():>5}MiB | {temp.strip():>2}°C'
    except Exception:
        return 'GPU n/a'


def smooth_estimate(out_dir, iters):
    """从 checkpoint 文件估: step, s/it, ETA。"""
    steps = checkpoint_steps(out_dir)
    if not steps:
        return None
    step = steps[-1][0]
    s_per_it = None
    if len(steps) >= 2:
        dt = steps[-1][1] - steps[-2][1]
        dstep = steps[-1][0] - steps[-2][0]
        if dstep > 0:
            s_per_it = dt / dstep
    return step, s_per_it


def bar(step, iters, s_per_it):
    pct = min(100, step / iters * 100)
    w = 30
    filled = int(pct / 100 * w)
    speed = f'{s_per_it:.2f}s/it' if s_per_it else '?'
    eta = f'{(iters-step)*s_per_it/60:.0f}min' if (s_per_it and step > 0) else '?'
    return (f'\n  训练 [{ "#" * filled}{"-" * (w - filled)}] '
            f'{step:>5}/{iters} {pct:5.1f}%  {speed}  ETA {eta}')


def snapshot(out_dir, iters):
    lines = []
    # 平滑日志优先
    pf = read_progress_file(out_dir)
    if pf:
        last = pf[-1]
        m = re.match(r'step\s+(\d+)', last)
        if m:
            step = int(m.group(1))
            lines.append(bar(step, iters, None))
            lines.append('  最近日志: ' + last)
        else:
            lines.append('  (train_progress.txt 为空)')
    else:
        est = smooth_estimate(out_dir, iters)
        if est:
            step, sp = est
            lines.append(bar(step, iters, sp))
        else:
            lines.append(f'  (未找到 checkpoint 于 {out_dir})')
    lines.insert(0, '  ' + gpu_info())
    return '\n'.join(lines)


def main():
    once = len(os.sys.argv) > 1 and os.sys.argv[1] == 'once'
    if once:
        os.sys.argv = [os.sys.argv[0]] + os.sys.argv[2:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='D:/models/runs/v3plus_v2')
    ap.add_argument('--iters', type=int, default=12000)
    args = ap.parse_args()

    if once:
        print(snapshot(args.dir, args.iters))
        return
    try:
        while True:
            print('\033[2J\033[H' + snapshot(args.dir, args.iters), end='', flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        print('\nbye')


if __name__ == '__main__':
    main()
