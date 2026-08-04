#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v3 100 张测试集生成进度条。

数据源:
  - D:/models/v3_test100_out/ 下 jpg 计数 (已完成张数)
  - 后台任务日志尾部 (当前张)
  - nvidia-smi GPU 状态

用法:
  python v3_progress.py          # 动画 (每3秒刷新, Ctrl+C 退出)
  python v3_progress.py once     # 一次性快照
"""
import os
import re
import sys
import time
import glob
import subprocess

OUT_DIR = 'D:/models/v3_test100_out'
LOG = r'C:/Users/YUKINA~1/AppData/Local/Temp/claude/C-----------------/0f1d62a5-0f5b-4a6a-b74f-19fb74c60224/tasks/boq56drrg.output'
TOTAL = 100


def done_count():
    return len(glob.glob(os.path.join(OUT_DIR, '*.jpg')))


def current_info():
    if not os.path.exists(LOG):
        return None
    size = os.path.getsize(LOG)
    with open(LOG, 'rb') as f:
        f.seek(max(0, size - 1500))
        tail = f.read().decode('utf-8', errors='replace')
    m = re.findall(r'\[(\d+)/\d+\]', tail)
    if m:
        return int(m[-1])
    return None


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
    return (f'v3出图 [{ "#" * filled}{"-" * (w - filled)}] '
            f'{done:>3}/{TOTAL} {pct:5.1f}%')


def snapshot():
    done = done_count()
    cur = current_info()
    lines = ['  ' + gpu_info(), bar(done)]
    if cur and done < TOTAL:
        # 每张约 40 秒 (37张约25分钟)
        est = (TOTAL - done) * 0.7
        lines.append(f'  当前: [{cur}/100] | 预计剩余 {est:.0f} 分钟')
    return '\n'.join(lines)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'once':
        print(snapshot())
        sys.exit(0)
    try:
        while True:
            print('\033[2J\033[H' + snapshot(), end='', flush=True)
            time.sleep(3)
    except KeyboardInterrupt:
        print('\nbye')
