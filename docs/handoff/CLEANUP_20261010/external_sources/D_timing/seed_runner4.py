# -*- coding: utf-8 -*-
"""VOSR 种子渲染 v4 (断点续跑): 剩余种子 [4,5,6,8,9,10,11], 已完成的种子自动跳过 (幂等)
带单实例锁: 已有实例运行时直接退出 (防 auto_pipeline 与手动启动撞车)
用法: D:/Python/python.exe seed_runner4.py
"""
import ctypes
import ctypes.wintypes
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['TORCHDYNAMO_SUPPRESS_ERRORS'] = '1'
CWD = r'C:/models/finals'
SCRIPT = r'D:/models/finals/VOSR/inference_vosr_onestep.py'
CKPT = r'preset/ckpts/VOSR_0.5B_os'
TEST = r'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'
SEEDS = [4, 5, 6, 8, 9, 10, 11]
LOCK = Path(r'D:/tmp_timing/seed_runner4.lock')


def pid_alive(pid):
    if pid <= 0:
        return False
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD]
    kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
    h = kernel32.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    kernel32.CloseHandle(h)
    return True


def acquire_lock():
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
        except Exception:
            old = 0
        if old and pid_alive(old):
            print(f'[lock] 已有实例 pid={old} 运行中, 退出', flush=True)
            sys.exit(0)
    LOCK.write_text(str(os.getpid()))
    print(f'[lock] acquired pid={os.getpid()}', flush=True)


def seed_ok(seed):
    d = Path(CWD) / f'preset/vosr_seed{seed}_test'
    return d.is_dir() and len(list(d.glob('*.png'))) >= 100


def main():
    acquire_lock()
    try:
        for seed in SEEDS:
            if seed_ok(seed):
                print(f'[seed {seed}] 已完成, 跳过', flush=True)
                continue
            out = f'preset/vosr_seed{seed}_test'
            cmd = [r'D:/Python/python.exe', SCRIPT, '-c', CKPT, '-i', TEST, '-o', out,
                   '-u', '1', '--tile_size', '512', '--vae_tile_size', '1024',
                   '--vae_tile_overlap', '64', '--seed', str(seed)]
            print(f'[seed {seed}] start {time.strftime("%H:%M:%S")}', flush=True)
            t0 = time.time()
            subprocess.run(cmd, cwd=CWD, check=False)
            print(f'[seed {seed}] done {time.strftime("%H:%M:%S")} ({time.time() - t0:.0f}s)', flush=True)
        print('[all seeds done]', flush=True)
    finally:
        if LOCK.exists():
            LOCK.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
