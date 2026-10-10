# -*- coding: utf-8 -*-
"""上传 div2k pngs tar (3.29GB) 到 P100 服务器 + 解包校验 (2026-09-04)
用法: python p100_upload_div2k.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r'D:/tmp_timing')
from p100_upload import connect

LOCAL = Path(r'D:/tmp_timing/div2k_pngs.tar')
REMOTE = 'p100/data/div2k_pngs.tar'
LOG = Path(r'D:/tmp_timing/upload_div2k.log')


def log(m):
    line = time.strftime('[%m-%d %H:%M:%S] ') + m
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    t0 = time.time()
    last = [0]

    def cb(done, total):
        if done - last[0] >= 200 * 1024 * 1024 or done == total:
            log(f'  progress {done/1e9:.2f}/{total/1e9:.2f} GB ({done/total*100:.0f}%)')
            last[0] = done

    ok = False
    for attempt in range(1, 6):
        try:
            cli = connect()
            sftp = cli.open_sftp()
            log(f'[upload] attempt {attempt}: {LOCAL.name} ({LOCAL.stat().st_size/1e9:.2f} GB)')
            sftp.put(str(LOCAL), REMOTE, callback=cb)
            sftp.close()
            cli.close()
            ok = True
            break
        except Exception as e:
            log(f'[warn] attempt {attempt} 失败: {str(e)[:150]}')
            time.sleep(10)
    if not ok:
        log('[CRITICAL] 上传 5 次失败, 人工介入')
        sys.exit(1)
    # 校验大小
    cli = connect()
    _i, _o, _e = cli.exec_command(f'ls -la ~/{REMOTE}', timeout=30)
    out = _o.read().decode()
    log(f'[verify] 远端: {out.strip()}')
    # 解包到 pngtxt_ori (与现有 20 文件冒烟子集合并)
    _i, _o, _e = cli.exec_command(
        'mkdir -p ~/p100/data/pngtxt_ori && cd ~/p100/data/pngtxt_ori && '
        'tar -xf ~/p100/data/div2k_pngs.tar && '
        f'echo PNG_COUNT=$(ls *.png | wc -l) && rm ~/p100/data/div2k_pngs.tar && echo UNTAR_OK',
        timeout=600)
    out = _o.read().decode('utf-8', 'replace')
    err = _e.read().decode('utf-8', 'replace')
    log(f'[untar] {out.strip()}')
    if err.strip():
        log(f'[stderr] {err[-300:]}')
    cli.close()
    log(f'[done] 上传+解包完成, 耗时 {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
