# -*- coding: utf-8 -*-
"""w48 训练接力器: 等 train_w48.log 出现 [done] 后自动启动 w48 测试集直选 sweep
(s10k-16k net/ema 14 件, 渲染输出仍写 D:/models/sweep_test_out)
启动: Start-Process D:/Python/python.exe -ArgumentList 'D:/tmp_timing/watcher_w48.py' -WindowStyle Hidden
"""
import subprocess
import time

LOG = r'D:/tmp_timing/train_w48.log'
OUT = r'D:/tmp_timing/sweep_w48.log'
MARK = r'D:/tmp_timing/watcher_w48.marker'


def main():
    try:
        if '[all done]' in open(OUT, encoding='utf-8', errors='ignore').read():
            print('sweep already finished, exit', flush=True)
            return
    except Exception:
        pass
    while True:
        try:
            txt = open(LOG, encoding='utf-8', errors='ignore').read()
        except Exception:
            txt = ''
        if '[done]' in txt:
            break
        time.sleep(120)
    subprocess.Popen(
        [r'D:/Python/python.exe', r'D:/tmp_timing/sweep_test_all.py', '--ckpt-dirs', r'C:/models/metric_net_w48'],
        cwd=r'D:/tmp_timing',
        stdout=open(OUT, 'w'), stderr=subprocess.STDOUT,
        creationflags=0x00000008 | 0x00000200)
    with open(MARK, 'w') as f:
        f.write('sweep launched')
    print('sweep launched', flush=True)


if __name__ == '__main__':
    main()
