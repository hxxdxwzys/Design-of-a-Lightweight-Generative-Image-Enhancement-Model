#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通过 exec_command 的 stdin 流式上传文件(不走 SFTP)。
用法: python ssh_cat_upload.py <本地路径> <远程路径>
环境变量: RHOST / RPORT(默认22) / RUSER(默认root) / RPASS
"""
import os
import sys
import paramiko

host = os.environ["RHOST"]
port = int(os.environ.get("RPORT", "22"))
user = os.environ.get("RUSER", "root")
pw = os.environ["RPASS"]
local, remote = sys.argv[1], sys.argv[2]

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(host, port=port, username=user, password=pw, timeout=20,
            allow_agent=False, look_for_keys=False)
cli.get_transport().set_keepalive(10)

stdin, stdout, stderr = cli.exec_command(f"cat > {remote}", bufsize=-1)
with open(local, "rb") as f:
    while True:
        data = f.read(65536)
        if not data:
            break
        stdin.write(data)
        stdin.flush()
stdin.channel.shutdown_write()
code = stdout.channel.recv_exit_status()
err = stderr.read().decode("utf-8", "replace")
cli.close()
print(f"uploaded {os.path.getsize(local)} bytes -> {remote} [exit {code}] {err[:200]}")
