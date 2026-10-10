#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DiffBIR 100 张出图进度条。

数据源:
  - D:/models/diffbir/test100_run2.log 里 "save result" 计数 (已完成张数)
  - 日志尾部进度条 (当前张的 tiled 阶段)
  - nvidia-smi GPU 状态

用法:
  python diffbir_progress.py          # 动画 (每5秒刷新, Ctrl+C 退出)
  python diffbir_progress.py once     # 一次性快照
"""
import os
import re
import sys
import time
import subprocess

LOG = 'D:/models/diffbir/test100_run2.log'
TOTAL = 100
LOG2 = 'D:/models/diffbir/val4k_s10.log'  # steps10 验证集 (3张)
TOTAL2 = 3


def done_count():
    if not os.path.exists(LOG):
        return 0
    with open(LOG, 'r', encoding='utf-8', errors='replace') as f:
        return f.read().count('save result')


def current_tile_progress():
    """从日志尾部抓当前 tiled 阶段进度 (x/y)。"""
    if not os.path.exists(LOG):
        return None, None
    size = os.path.getsize(LOG)
    with open(LOG, 'rb') as f:
        f.seek(max(0, size - 3000))
        tail = f.read().decode('utf-8', errors='replace')
    # 最后出现的 "n/m" 进度
    matches = re.findall(r'(\d+)/(\d+)', tail.replace('\r', '\n'))
    if matches:
        return int(matches[-1][0]), int(matches[-1][1])
    return None, None


def gpu_info():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10)
        util, mem, temp = r.stdout.strip().split(',')
        return f'GPU {util.strip():>3}% | {mem.strip():>5}MiB | {temp.strip():>2}C'
    except Exception:
        return 'GPU n/a'


def bar(done):
    pct = min(100, done / TOTAL * 100)
    w = 30
    filled = int(pct / 100 * w)
    return (f'出图 [{ "#" * filled}{"-" * (w - filled)}] '
            f'{done:>3}/{TOTAL} {pct:5.1f}%')


def snapshot():
    lines = ['  ' + gpu_info()]
    # steps4 100张 (主任务)
    done = done_count()
    lines.append(bar(done))
    # steps10 验证集 (3张, 若有进行)
    if os.path.exists(LOG2):
        d2 = 0
        with open(LOG2, 'r', encoding='utf-8', errors='replace') as f:
            d2 = f.read().count('save result')
        lines.append(f'  steps10验证集 [{ "#" * min(30, d2)}{"-" * (30 - min(30, d2))}] '
                     f'{d2}/{TOTAL2}')
    cur, tot = current_tile_progress()
    if cur is not None and tot and done < TOTAL:
        frac = cur / tot if tot else 0
        est_min = (TOTAL - done - frac) * 3
        lines.append(f'  当前张: tile {cur}/{tot} | 预计剩余 {est_min:.0f} 分钟')
    return '\n'.join(lines)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'once':
        print(snapshot())
        sys.exit(0)
    try:
        while True:
            print('\033[2J\033[H' + snapshot(), end='', flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print('\nbye')
