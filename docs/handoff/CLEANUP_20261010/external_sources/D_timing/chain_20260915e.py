# -*- coding: utf-8 -*-
"""2026-09-15 链 E: 512 域 (down=1) GAN 学生两个点
动机(链 C 读数): 官方质量项≈"观感(无参)", 而非有参保真 —— 同一速度档下 pkgv7 四项有参全部劣于 pkgv5,
                却官方 +8.70。=> 目标 = 在 >=50x 下把"观感代理(clipiqa)"推上去。
瓶颈假设: v7gan 走 down=2 (512 tile 降到 256 再升回 512), 输出无法承载高频纹理 => clipiqa 只有 0.51。
v9b: 512 域, ch 32,32,32 (~0.3M, 算力~1.4x v7gan => 预计 ~40x)  —— 近同速的分辨率单变量
v9a: 512 域, ch 32,32,32,64 (0.87M, 算力~4x => 预计 ~15x)      —— 分辨率+容量
注意: UNet2DModel 默认 norm_num_groups=32 => 通道数必须能被 32 整除 (16 会报错, 已踩)
日志: D:/tmp_timing/chain_20260915e.log
"""
import subprocess, sys, time
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
import gan_chain as G

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'chain_20260915e.log'


def log(m):
    s = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


G.LOG = LOG
G.log = log

G.run('train v9b (512域 GAN, 0.3M, bs4)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '25000',
                                           '--bs', '4', '--res', '512', '--ch', '32,32,32',
                                           '--tag', 'v9b'], timeout=21600)
ck = Path(r'D:/models/tiny_v1/v9b_s25000.pt')
if ck.exists():
    G.productize(ck, 'TinyV9b', 1, '0.3M GAN蒸馏 512域 down=1 (近同速分辨率实验)')

G.run('train v9a (512域 GAN, 0.87M, bs4)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '25000',
                                            '--bs', '4', '--res', '512', '--ch', '32,32,32,64',
                                            '--tag', 'v9a'], timeout=36000)
ck = Path(r'D:/models/tiny_v1/v9a_s25000.pt')
if ck.exists():
    G.productize(ck, 'TinyV9a', 1, '0.87M GAN蒸馏 512域 down=1')
log('CHAIN_E_DONE')
