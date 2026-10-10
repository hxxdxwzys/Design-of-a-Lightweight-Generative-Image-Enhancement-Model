#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SFTP 上传本地文件到服务器。用法: python ssh_upload.py <本地路径> <远程路径>
环境变量: RHOST / RPORT(默认22) / RUSER(默认root) / RPASS
"""
import os, sys, paramiko

host = os.environ["RHOST"]
port = int(os.environ.get("RPORT", "22"))
user = os.environ.get("RUSER", "root")
pw = os.environ["RPASS"]
local, remote = sys.argv[1], sys.argv[2]

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(host, port=port, username=user, password=pw, timeout=20,
            allow_agent=False, look_for_keys=False)
sftp = cli.open_sftp()
sftp.put(local, remote)
sftp.close()
cli.close()
print(f"uploaded {local} -> {remote}")
