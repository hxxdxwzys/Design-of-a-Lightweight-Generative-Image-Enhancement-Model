# -*- coding: utf-8 -*-
"""P100 教学服务器验收探针 (2026-09-01): 连接 bigdata.dynu.net:9077 收集环境信息
用法: D:/Python/python.exe p100_probe.py
"""
import socket
import sys
import time

import paramiko

HOST, PORT = 'bigdata.dynu.net', 9077
USER, PWD = 'tempuser', 'temp76'

CMDS = [
    ('系统', 'uname -a && cat /etc/os-release | head -2'),
    ('GPU', 'nvidia-smi'),
    ('Python', 'python3 --version 2>&1; which python3'),
    ('磁盘', 'df -h / 2>/dev/null | tail -1; df -h /home 2>/dev/null | tail -1'),
    ('内存', 'free -g | head -2'),
    ('CPU', 'nproc; lscpu | grep "Model name" | head -1'),
    ('用户', 'whoami; sudo -n true 2>&1 && echo HAS_SUDO || echo NO_SUDO'),
    ('网络-外网', 'curl -sI -m 8 https://hf-mirror.com 2>&1 | head -1; curl -sI -m 8 https://www.modelscope.cn 2>&1 | head -1; curl -sI -m 8 https://download.pytorch.org 2>&1 | head -1'),
]


def main():
    print(f'[probe] {HOST}:{PORT} ({USER})', flush=True)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    last_err = None
    for attempt in range(3):
        try:
            cli.connect(HOST, port=PORT, username=USER, password=PWD,
                        timeout=30, banner_timeout=30, auth_timeout=30)
            break
        except Exception as e:
            last_err = e
            print(f'[warn] 连接失败 attempt {attempt + 1}: {e}', flush=True)
            time.sleep(5)
    else:
        print(f'[fatal] 连不上: {last_err}', flush=True)
        sys.exit(1)
    print('[ok] 已连接', flush=True)
    for name, cmd in CMDS:
        print(f'\n===== {name} =====', flush=True)
        try:
            _in, _out, _err = cli.exec_command(cmd, timeout=60)
            out = _out.read().decode('utf-8', errors='replace').strip()
            err = _err.read().decode('utf-8', errors='replace').strip()
            print(out if out else '(空)')
            if err:
                print(f'[stderr] {err[:500]}')
        except Exception as e:
            print(f'[err] {e}')
    cli.close()
    print('\n[done]', flush=True)


if __name__ == '__main__':
    main()
