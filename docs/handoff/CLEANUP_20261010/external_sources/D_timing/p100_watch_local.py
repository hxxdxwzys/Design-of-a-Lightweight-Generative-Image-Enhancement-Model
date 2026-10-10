# -*- coding: utf-8 -*-
"""本地自动链守护 (2026-09-03): 监控 autoval14b / autoval, 卡死或死亡自动重启
判定: 心跳文件新鲜度 (<6min) + 进程存活; 异常则杀+重启
用法: D:/Python/python.exe p100_watch_local.py
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r'D:/tmp_timing')

OUT_DIR = Path(r'D:/tmp_timing/p100_val')
WATCHERS = [
    {
        'name': 'autoval14',
        'script': r'D:/tmp_timing/p100_autoval_14b.py',
        'hb': OUT_DIR / 'autoval14.hb',
        'cwd': r'C:/models/finals',
        'log': OUT_DIR / 'autoval14.log',
    },
]
POLL = 120
HB_FRESH = 360  # 心跳新鲜阈值 6 分钟
LIVE_LOG = Path(r'D:/tmp_timing/watch_local.log')


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LIVE_LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def find_pid(script):
    """按命令行匹配进程 pid (Windows)"""
    out = subprocess.run(
        ['powershell', '-NoProfile', '-Command',
         f"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         f"Where-Object {{ $_.CommandLine -like '*{os.path.basename(script)}*' }} | "
         f"Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True, timeout=60)
    pids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    return pids


def start_watcher(w):
    log(f'[start] {w["name"]}')
    with open(w['log'], 'a', encoding='utf-8') as f:
        subprocess.Popen([sys.executable, w['script']], cwd=w['cwd'],
                         stdout=f, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def gpu_busy():
    """GPU 活跃判定: 利用率>30% 或 显存>2GB 视为渲染中"""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                              '--format=csv,noheader,nounits'], capture_output=True,
                             text=True, timeout=30)
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 2 and (int(parts[0]) > 30 or int(parts[1]) > 2000):
                return True
    except Exception:
        pass
    return False


def main():
    log(f'[boot] 本地守护启动, poll={POLL}s (健康=进程在 AND (心跳新 OR GPU忙))')
    while True:
        for w in WATCHERS:
            pids = find_pid(w['script'])
            hb_ok = w['hb'].exists() and (time.time() - w['hb'].stat().st_mtime) < HB_FRESH
            busy = gpu_busy()
            if pids and (hb_ok or busy):
                pass  # 健康 (心跳新, 或渲染中 GPU 忙)
            elif pids and not hb_ok and not busy:
                log(f'[卡死] {w["name"]} 心跳过期且 GPU 空闲, 杀 {pids} 重启')
                for p in pids:
                    subprocess.run(['taskkill', '/PID', str(p), '/F'], capture_output=True)
                time.sleep(3)
                start_watcher(w)
            elif not pids:
                log(f'[死亡] {w["name"]} 进程不存在 -> 重启')
                start_watcher(w)
        time.sleep(POLL)


if __name__ == '__main__':
    main()
