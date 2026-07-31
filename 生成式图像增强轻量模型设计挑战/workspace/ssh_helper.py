#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通过 paramiko 用密码 SSH 执行远程命令并流式回显。
环境变量: RHOST / RPORT(默认22) / RUSER(默认root) / RPASS / RTIMEOUT(默认1200s)
用法: python ssh_helper.py '<远程命令>'
     或: echo '<命令>' | python ssh_helper.py
"""
import os, sys, time, paramiko

host = os.environ["RHOST"]
port = int(os.environ.get("RPORT", "22"))
user = os.environ.get("RUSER", "root")
pw = os.environ["RPASS"]
cmd = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
timeout = int(os.environ.get("RTIMEOUT", "1200"))

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
last = None
for _attempt in range(6):
    try:
        cli.connect(host, port=port, username=user, password=pw, timeout=20,
                    allow_agent=False, look_for_keys=False)
        last = None
        break
    except Exception as e:
        last = e
        import time as _t
        _t.sleep(3)
if last is not None:
    raise last
chan = cli.get_transport().open_session()
chan.settimeout(timeout)
chan.exec_command(cmd)
while not chan.exit_status_ready():
    if chan.recv_ready():
        sys.stdout.buffer.write(chan.recv(4096)); sys.stdout.buffer.flush()
    if chan.recv_stderr_ready():
        sys.stderr.buffer.write(chan.recv_stderr(4096)); sys.stderr.buffer.flush()
    time.sleep(0.03)
while chan.recv_ready():
    sys.stdout.buffer.write(chan.recv(4096)); sys.stdout.buffer.flush()
while chan.recv_stderr_ready():
    sys.stderr.buffer.write(chan.recv_stderr(4096)); sys.stderr.buffer.flush()
code = chan.recv_exit_status()
cli.close()
sys.exit(code)
