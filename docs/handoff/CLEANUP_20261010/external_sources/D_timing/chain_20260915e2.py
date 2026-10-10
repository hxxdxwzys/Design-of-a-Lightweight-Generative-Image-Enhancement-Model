# -*- coding: utf-8 -*-
"""2026-09-15 夜间链 E2: 512 域分辨率实验 + 自动分支
槽1 (必跑): v9b = 512 域, ch 32,32,32 (0.38M), GAN, 25k 步, bs4  => ~4h  (down=1, 近同速分辨率单变量)
   跑完 productize -> measure test30 (metrics_v9b.json)
槽2 (自动分支, 判据预先登记在此文件里):
   若 v9b test30 clipiqa >= 0.58  => 分辨率有效 => v9a = 512 域 ch 32,32,32,64 (0.87M), 12k 步 (~7.7h? 太长)
                                     => 改用 8k 步 (~5h), down=1
   否则                          => v8s = 256 域 ch 32,32,32,64 (0.87M), w_gan 0.3 (加强对抗), 40k 步 (~3h)
日志: D:/tmp_timing/chain_20260915e2.log
"""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
import gan_chain as G

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'chain_20260915e2.log'


def log(m):
    s = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


G.LOG = LOG
G.log = log
CK = Path(r'D:/models/tiny_v1')

# ---------------- 槽 1: v9b ----------------
G.run('train v9b (512域 GAN 0.38M bs4 25k)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '25000',
                                              '--bs', '4', '--res', '512', '--ch', '32,32,32', '--tag', 'v9b'],
      timeout=21600)
ck = CK / 'v9b_s25000.pt'
if ck.exists():
    G.productize(ck, 'TinyV9b', 1, '0.38M GAN蒸馏 512域 down=1 (分辨率实验)')

mj = Path(r'D:/models/pkg_TinyV9b/test30_metrics.json')
cq = None
if mj.exists():
    cq = float(json.loads(mj.read_text(encoding='utf-8')).get('clipiqa', 0))
log('[branch] v9b test30 clipiqa = %s (判据 0.58)' % cq)

# ---------------- 槽 2: 自动分支 ----------------
if cq is not None and cq >= 0.58:
    log('[branch] 分辨率有效 -> v9a (512域, 0.87M, 8k 步)')
    G.run('train v9a (512域 GAN 0.87M bs4 8k)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '8000',
                                                 '--bs', '4', '--res', '512', '--ch', '32,32,32,64',
                                                 '--tag', 'v9a'], timeout=28800)
    ck = CK / 'v9a_s8000.pt'
    if ck.exists():
        G.productize(ck, 'TinyV9a', 1, '0.87M GAN蒸馏 512域 down=1 (8k 步)')
else:
    log('[branch] 分辨率不足以单独解决 -> v8s (256域, 0.87M, w_gan 0.3, 40k 步)')
    G.run('train v8s (256域 GAN w0.3 40k)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '40000',
                                             '--bs', '8', '--ch', '32,32,32,64', '--w_gan', '0.3',
                                             '--tag', 'v8s'], timeout=21600)
    ck = CK / 'v8s_s40000.pt'
    if ck.exists():
        G.productize(ck, 'TinyV8s', 2, '0.87M GAN w_gan=0.3 40k 步 256域')
log('CHAIN_E2_DONE')
