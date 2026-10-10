# -*- coding: utf-8 -*-
"""2026-09-15 链 D: v9a = 512 域 GAN 学生 (ch 16,16,16,32, 与 0.87M@256 同算力 => 速度不变)
动机: 256 域把 512 tile 先降采样 2x => 学生看不到高频细节, 只能幻觉; 官方 LQ 本身只轻退化.
      保留 GAN 观感 (已证明活的杠杆), 只把输入分辨率改回 1x, 算力/速度保持不变 => 干净的"分辨率"单变量。
日志: D:/tmp_timing/chain_20260915d.log
"""
import subprocess, sys, time
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
import gan_chain as G

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'chain_20260915d.log'


def log(m):
    s = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


G.LOG = LOG
G.log = log

G.run('train v9a (512域 GAN, 0.22M, bs4)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '25000',
                                            '--bs', '4', '--res', '512', '--ch', '16,16,16,32',
                                            '--tag', 'v9a'])
ck = Path(r'D:/models/tiny_v1/v9a_s25000.pt')
if ck.exists():
    G.productize(ck, 'TinyV9a', 1, '0.22M GAN蒸馏 512域 (down=1, 与 0.87M@256 同算力)')
log('CHAIN_D_DONE')
