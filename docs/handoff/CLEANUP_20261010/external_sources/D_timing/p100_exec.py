# -*- coding: utf-8 -*-
"""P100 服务器通用命令执行器: python p100_exec.py "命令" [--timeout 60]
入口: 环境变量 P100_HOST/P100_PORT 覆盖 (默认校园网隧道 10.96.138.171:2222)
"""
import os
import sys
import time

import paramiko

HOST = os.environ.get('P100_HOST', '10.96.138.171')
PORT = int(os.environ.get('P100_PORT', '2222'))
USER, PWD = 'tempuser', 'temp76'


def main():
    cmd = sys.argv[1]
    timeout = 120
    if '--timeout' in sys.argv:
        timeout = int(sys.argv[sys.argv.index('--timeout') + 1])
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for i in range(3):
        try:
            cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
            break
        except Exception as e:
            print(f'[warn] 连接失败 {i + 1}: {e}', flush=True)
            time.sleep(5)
    _in, _out, _err = cli.exec_command(cmd, timeout=timeout)
    out = _out.read().decode('utf-8', errors='replace')
    err = _err.read().decode('utf-8', errors='replace')
    if out:
        print(out, flush=True)
    if err:
        print(f'[stderr]\n{err}', flush=True)
    cli.close()


if __name__ == '__main__':
    main()
