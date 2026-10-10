#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通过 exec `cat remote` 流式下载文件(不走 SFTP)。
用法: python ssh_cat_download.py <远程路径> <本地路径>
环境变量: RHOST / RPORT(默认22) / RUSER(默认root) / RPASS
注意: 调用时需 export MSYS2_ARG_CONV_EXCL='*' 以免 /data/... 被转成 Windows 路径。
"""
import os
import sys
import paramiko

host = os.environ["RHOST"]
port = int(os.environ.get("RPORT", "22"))
user = os.environ.get("RUSER", "root")
pw = os.environ["RPASS"]
remote, local = sys.argv[1], sys.argv[2]

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(host, port=port, username=user, password=pw, timeout=20,
            allow_agent=False, look_for_keys=False)
cli.get_transport().set_keepalive(10)
stdin, stdout, stderr = cli.exec_command(f"cat {remote}", bufsize=-1)
with open(local, "wb") as f:
    while True:
        data = stdout.read(65536)
        if not data:
            break
        f.write(data)
err = stderr.read().decode("utf-8", "replace")
code = stdout.channel.recv_exit_status()
cli.close()
print(f"downloaded {remote} -> {local} ({os.path.getsize(local)} bytes) [exit {code}] {err[:200]}")
