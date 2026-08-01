#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地包装: 从项目根 .secrets 读服务器凭据, 执行远程命令/上传/下载。

用法:
    python srv.py run '<remote cmd>'        # 远程执行 (同 ssh_helper)
    python srv.py up <local> <remote>       # 上传 (exec+cat, 绕开SFTP)
    python srv.py down <remote> <local>     # 下载
"""
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = _here
while ROOT and not os.path.exists(os.path.join(ROOT, '.secrets')):
    ROOT = os.path.dirname(ROOT)


def load_secrets():
    sec = {}
    with open(os.path.join(ROOT, '.secrets'), encoding='utf-8') as f:
        for line in f:
            line = line.split('#', 1)[0]  # 去掉行尾注释
            m = re.match(r'\s*(\w+)\s*=\s*(.*?)\s*$', line)
            if m:
                sec[m.group(1)] = m.group(2).strip()
    return sec


def main():
    sec = load_secrets()
    os.environ['RHOST'] = sec['公网IP']
    os.environ['RPORT'] = sec.get('SSH端口', '22')
    os.environ['RUSER'] = sec.get('用户', 'root')
    os.environ['RPASS'] = sec['密码']

    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if mode == 'run':
        cmd = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
        rc = subprocess.call([sys.executable, os.path.join(here, 'ssh_helper.py'), cmd])
        sys.exit(rc)
    elif mode == 'up':
        local, remote = sys.argv[2], sys.argv[3]
        sys.exit(subprocess.call(
            [sys.executable, os.path.join(here, 'ssh_cat_upload.py'), local, remote]))
    elif mode == 'down':
        remote, local = sys.argv[2], sys.argv[3]
        sys.exit(subprocess.call(
            [sys.executable, os.path.join(here, 'ssh_cat_download.py'), remote, local]))
    else:
        print('unknown mode', mode)
        sys.exit(2)


if __name__ == '__main__':
    main()
