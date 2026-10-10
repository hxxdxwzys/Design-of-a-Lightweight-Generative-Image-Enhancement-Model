# -*- coding: utf-8 -*-
"""磁盘水位守卫：任何写盘阶段开始前调用，低于阈值即拒绝启动（fail-closed）。

用法: python disk_guard.py <path> <required_gib> [<label>]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def free_gib(path) -> float:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(str(p)).free / (1024 ** 3)


def require_free(path, required_gib, label=''):
    got = free_gib(path)
    if got < float(required_gib):
        raise SystemExit('DISK_GUARD_FAIL %s: free=%.2f GiB < required=%.2f GiB (%s)'
                         % (path, got, float(required_gib), label))
    return got


if __name__ == '__main__':
    p = sys.argv[1]
    need = float(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else ''
    got = require_free(p, need, label)
    print('DISK_GUARD_OK free=%.2f GiB required=%.2f GiB path=%s' % (got, need, p))
