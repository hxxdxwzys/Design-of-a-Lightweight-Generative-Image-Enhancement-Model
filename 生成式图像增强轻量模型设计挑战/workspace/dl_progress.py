#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三个下载任务的实时进度条 (每2秒刷新, Ctrl+C 退出)。
用法: python dl_progress.py
或一次性查看: python dl_progress.py once
"""
import os
import sys
import glob
import time

# (名称, 路径, 目标字节, 类型: file/dir/segments)
TARGETS = [
    ('torch',   'D:/models/torch_210_cu126_win.whl', 2589881709, 'segments'),
    ('div2k',   'D:/datasets/div2k.zip',             3818795520, 'file'),
    ('sd21base', 'D:/models/sd21base',                4600000000, 'dir'),
]


def get_size(kind, path):
    if kind == 'segments':
        total = 0
        for f in glob.glob(os.path.join(os.environ.get('TMP',
                         r'C:\Users\Yuki Nagato\AppData\Local\Temp'),
                         'dl_segs_*', '*.part')):
            total += os.path.getsize(f)
        return total
    if kind == 'dir':
        t = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    t += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return t
    if os.path.exists(path):
        return os.path.getsize(path)
    return 0


def bar(name, cur, total):
    pct = min(100, cur / total * 100)
    w = 24
    filled = int(pct / 100 * w)
    return (f'{name:<9} [{"#" * filled}{"-" * (w - filled)}] '
            f'{cur/1e6:>6.0f}/{total/1e6:.0f}MB {pct:5.1f}%')


def snapshot():
    return [bar(n, get_size(k, p), t) for n, p, t, k in TARGETS]


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'once':
        for line in snapshot():
            print(line)
        sys.exit(0)
    try:
        while True:
            print('\033[2J\033[H' + '\n'.join(snapshot()), end='', flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        print('\nbye')
