# -*- coding: utf-8 -*-
"""链 M (2026-09-15 下午) — 合规"高速档"搜索: 用 v11b 的 512 域监督技巧压到更粗的域

新分数模型 (2026-09-15, 见 HANDOFF 22): score = Q(图) x (r/0.78)^0.18
  => 速度是幂律(不饱和), 而 Q 的下限 ≈ 16.6 (LQ 档图也能拿 ~16.6)
  => 合规最优 = "不劣化图像的前提下尽量快":  v4g(128域, 178.6x, look 0.279) 预测 44~49
本链: v16 = res 64 (down=8) + 512 域损失 + band-limit D + GAN; v17 = res 128 (down=4) 同配方
判据(机械): ① test30 clipiqa >= 0.30 (确实在做增强, 不是空模型/劣化) ② 空闲 >= 100x
            ③ 预测分 > 44  => 值得占一发
日志: D:/tmp_timing/chain_20260915m.log
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
import gan_chain as G

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
CK = Path('D:/models/tiny_v1')
LOG = T / 'chain_20260915m.log'


def log(m):
    s = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


def run(name, cmd, timeout=28800):
    log('>>> ' + name)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=timeout)
        rc, out = p.returncode, (p.stdout or '') + (p.stderr or '')
    except subprocess.TimeoutExpired as e:
        rc, out = 124, 'TIMEOUT %s\n%s' % (timeout, e)
    except Exception as e:
        rc, out = 125, 'EXC %s: %s' % (type(e).__name__, e)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(out + '\n')
    log('<<< %s rc=%d (%.1f min)' % (name, rc, (time.time() - t0) / 60))
    return rc, out


G.run = run
G.LOG = LOG
G.log = log
log('=== 链 M 启动 ===')

# v16: res 64 (down=8)
run('train v16 (res64 + 512域损失 + bandlimitD)',
    [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '20000', '--bs', '8', '--res', '64',
     '--loss_res', '512', '--d_band', '1', '--ch', '32,32,32,64', '--tag', 'v16',
     '--init_from', str(CK / 'v11b_s25000.pt')], timeout=21600)
if (CK / 'v16_s20000.pt').exists():
    G.productize(CK / 'v16_s20000.pt', 'TinyV16', 8, 'res64 + 512loss + bandlimitD')

# v17: res 128 (down=4)
run('train v17 (res128 + 512域损失 + bandlimitD)',
    [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '20000', '--bs', '6', '--res', '128',
     '--loss_res', '512', '--d_band', '1', '--ch', '32,32,32,64', '--tag', 'v17',
     '--init_from', str(CK / 'v16_s20000.pt')], timeout=21600)
if (CK / 'v17_s20000.pt').exists():
    G.productize(CK / 'v17_s20000.pt', 'TinyV17', 4, 'res128 + 512loss + bandlimitD')

log('=== 产物 ===')
for f in sorted(Path('D:/models').glob('pkg_TinyV1[67]/test30_metrics.json')):
    try:
        j = json.loads(f.read_text(encoding='utf-8'))
        log('%-14s clipiqa=%-8s niqe=%-8s n=%s' % (f.parent.name, j.get('clipiqa'), j.get('niqe'), j.get('n')))
    except Exception as e:
        log('!! %s %s' % (f, e))
log('CHAIN_M_DONE')
