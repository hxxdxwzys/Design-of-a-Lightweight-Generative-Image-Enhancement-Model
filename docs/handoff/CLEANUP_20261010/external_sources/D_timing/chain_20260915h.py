# -*- coding: utf-8 -*-
"""2026-09-15 链 H (单进程串行, 加固版; 不再用跨链等待, 消除"误判对方已死"的并发 bug)
依据新读数: v9b(512域) test30 clipiqa = 0.6409 @ 12.4~12.7x  => 分辨率假设成立, 观感过了 0.64 的饱和拐点。
   => 剩下唯一的大机会: 【在 79x 的 256 域学生上拿到 0.6+ 的观感】(预测 50~55)。
   H1: v11 = 256 域学生, 但 **损失/判别器放在 512 域** (学生学会抵消上采样模糊) —— 直击"上采样把高频糊掉"这个瓶颈
   H2: v9b2 = 512 域 0.38M, 45k 步 (已证明配方, 更长训练, 期望观感 0.67~0.70 @ 12.4x)
   H3: 闲置测量窗口 (原生分辨率探针 + 噪声注入探针)
日志: D:/tmp_timing/chain_20260915h.log
"""
import json, subprocess, sys, time, shutil
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
import gan_chain as G

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'chain_20260915h.log'
CK = Path('D:/models/tiny_v1')


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
        rc, out = 124, 'TIMEOUT %ss\n%s' % (timeout, e)
    except Exception as e:
        rc, out = 125, 'EXC %s: %s\n' % (type(e).__name__, e)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(out + '\n')
    log('<<< %s rc=%d (%.1f min)' % (name, rc, (time.time() - t0) / 60))
    return rc, out


def gpu_used():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=60)
        return int((p.stdout or '0').strip().splitlines()[0])
    except Exception:
        return -1


def wait_gpu_free(thresh=2200, need=3, ceiling_h=3):
    t0 = time.time(); ok = 0
    while time.time() - t0 < ceiling_h * 3600:
        u = gpu_used()
        ok = ok + 1 if 0 <= u < thresh else 0
        log('  [gpu] %s MiB (%d/%d)' % (u, ok, need))
        if ok >= need:
            return True
        time.sleep(60)
    return False


G.run = run
G.LOG = LOG
G.log = log

log('=== 链 H 启动 ===')
if not wait_gpu_free():
    log('!! GPU 不空闲, 仍继续 (训练优先)')

# ---------------- H1: 损失域 512 的 256 域学生 ----------------
run('train v11 (256域学生 + 512域损失, 25k)', [PY, '-u', str(T / 'train_tiny_gan.py'),
                                              '--steps', '25000', '--bs', '6', '--res', '256',
                                              '--loss_res', '512', '--ch', '32,32,32,64',
                                              '--tag', 'v11'], timeout=21600)
ck = CK / 'v11_s25000.pt'
if ck.exists():
    G.productize(ck, 'TinyV11', 2, '0.87M 256域学生 + 512域损失/GAN (抵消上采样模糊)')
else:
    log('!! v11 最终 ckpt 缺失, 跳过打包')

# ---------------- H2: v9b2 = 512 域更长训练 ----------------
run('train v9b2 (512域 0.38M 45k)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '45000',
                                     '--bs', '4', '--res', '512', '--ch', '32,32,32',
                                     '--tag', 'v9b2'], timeout=43200)
ck = CK / 'v9b2_s45000.pt'
if ck.exists():
    G.productize(ck, 'TinyV9b2', 1, '0.38M 512域 45k步 (v9b 配方加长)')
else:
    log('!! v9b2 最终 ckpt 缺失, 跳过打包')

# ---------------- H3: 测量窗口 ----------------
run('measure_probe_native30', [PY, '-u', str(T / 'measure_dir.py'), '--dir', 'D:/models/probe_native',
                               '--n', '30', '--metrics', 'niqe,clipiqa,lapvar', '--tag', 'probe_native30'])
for s in ('s0.050', 's0.100'):
    d = Path('D:/models/probe_noise') / s
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
run('render_noise', [PY, '-u', str(T / 'probe_noise.py'), '--ckpt', 'D:/models/tiny_v1/v7gan_s25000.pt',
                     '--out', 'D:/models/probe_noise', '--sigmas', '0.05,0.10', '--n', '10'])
for s in ('s0.000', 's0.050', 's0.100'):
    d = 'D:/models/probe_noise/' + s
    if Path(d).exists():
        run('measure_noise_' + s, [PY, '-u', str(T / 'measure_dir.py'), '--dir', d, '--n', '10',
                                   '--metrics', 'niqe,clipiqa,lapvar', '--tag', 'noise' + s.replace('.', '')])

log('=== 产物汇总 ===')
for f in sorted(Path('D:/models').glob('pkg_*/test30_metrics.json')):
    try:
        j = json.loads(f.read_text(encoding='utf-8'))
        log('%-18s clipiqa=%-8s niqe=%-8s n=%s' % (f.parent.name, j.get('clipiqa'), j.get('niqe'), j.get('n')))
    except Exception as e:
        log('!! %s %s' % (f, e))
log('CHAIN_H_DONE')
