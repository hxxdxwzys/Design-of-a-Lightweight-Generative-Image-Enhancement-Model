# -*- coding: utf-8 -*-
"""自动接力器: 等 train_v2b.log 出现 [done] 后, 自动启动 v2b 测试集直选 sweep
启动方式 (脱离会话):
  powershell Start-Process -FilePath 'D:\Python\python.exe' -ArgumentList 'D:/tmp_timing/watcher_v2b.py' -WindowStyle Hidden
"""
import subprocess
import time

LOG = r'D:/tmp_timing/train_v2b.log'
OUT = r'D:/tmp_timing/sweep_v2b.log'
MARK = r'D:/tmp_timing/watcher_v2b.marker'


def main():
    # 若 sweep 已在跑/已跑完, 直接退出 (防重复)
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
        time.sleep(90)
    # 训练完成 -> 启动 sweep (脱离父进程, 独立控制台)
    subprocess.Popen(
        [r'D:/Python/python.exe', r'D:/tmp_timing/sweep_test_all.py', '--ckpt-dirs', r'D:/models/metric_net_v2b'],
        cwd=r'D:/tmp_timing',
        stdout=open(OUT, 'w'), stderr=subprocess.STDOUT,
        creationflags=0x00000008 | 0x00000200)  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    with open(MARK, 'w') as f:
        f.write('sweep launched')
    print('sweep launched', flush=True)


if __name__ == '__main__':
    main()
