# -*- coding: utf-8 -*-
"""链 G (加固版): 等 F 结束 -> G1 加强对抗(同速) -> G2 按 v9b 分支(分辨率+容量 / 加宽)"""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
import gan_chain as G

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')


def log(m, LOG=None):
    s = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(s, flush=True)
    if LOG is not None:
        with LOG.open('a', encoding='utf-8') as f:
            f.write(s + '\n')


def safe_run(name, cmd, timeout=28800):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=timeout)
        rc, out = p.returncode, (p.stdout or '') + (p.stderr or '')
    except subprocess.TimeoutExpired as e:
        rc = 124
        out = 'TIMEOUT after %ss\n' % timeout + str(e)
    except Exception as e:
        rc = 125
        out = 'EXC %s: %s\n' % (type(e).__name__, e)
    return rc, out


def gpu_used():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=60)
        return int((p.stdout or '0').strip().splitlines()[0])
    except Exception:
        return -1


def wait_gpu_free(LOG, thresh=3200, need=3, ceiling_h=6):
    t0 = time.time(); ok = 0
    while time.time() - t0 < ceiling_h * 3600:
        u = gpu_used()
        ok = ok + 1 if 0 <= u < thresh else 0
        log('  [gpu] %s MiB (连续空闲 %d/%d)' % (u, ok, need), LOG)
        if ok >= need:
            return True
        time.sleep(60)
    return False


def stale(path, sec):
    try:
        return (time.time() - Path(path).stat().st_mtime) > sec
    except Exception:
        return True

LOG = T / 'chain_20260915g.log'
FLOG = T / 'chain_20260915f.log'
CK = Path('D:/models/tiny_v1')


def run(name, cmd, timeout=28800):
    log('>>> ' + name, LOG)
    t0 = time.time()
    rc, out = safe_run(name, cmd, timeout)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(out + '\n')
    log('<<< %s rc=%d (%.1f min)' % (name, rc, (time.time() - t0) / 60), LOG)
    return rc, out


G.run = run          # productize 内部也走加固版
G.LOG = LOG
G.log = lambda m: log(m, LOG)


def clip_of(pkg):
    f = Path('D:/models') / ('pkg_' + pkg) / 'test30_metrics.json'
    if f.exists():
        try:
            return float(json.loads(f.read_text(encoding='utf-8')).get('clipiqa', 0))
        except Exception:
            return None
    return None


log('等待测量窗口链 F (CHAIN_F_DONE) ...', LOG)
t0 = time.time()
while time.time() - t0 < 20 * 3600:
    txt = FLOG.read_text(encoding='utf-8', errors='replace') if FLOG.exists() else ''
    if 'CHAIN_F_DONE' in txt:
        break
    if time.time() - t0 > 3600 and stale(str(FLOG), 2400):
        log('!! F 疑似中断 (链日志 40 分钟未更新), 继续训练队列', LOG)
        break
    time.sleep(120)
log('F 结束/跳过 (等待 %.1f h), 开始训练队列' % ((time.time() - t0) / 3600), LOG)

if not wait_gpu_free(LOG):
    log('!! GPU 不空闲, 仍按队列继续 (可能与他人共用)', LOG)

v9b = clip_of('TinyV9b')
log('[读到] v9b=%s v9a=%s v8s=%s' % (v9b, clip_of('TinyV9a'), clip_of('TinyV8s')), LOG)

run('train v8t (256域 0.87M w_gan=0.25 45k)', [PY, '-u', str(T / 'train_tiny_gan.py'),
                                               '--steps', '45000', '--bs', '8', '--ch', '32,32,32,64',
                                               '--w_gan', '0.25', '--tag', 'v8t'], timeout=28800)
ck = CK / 'v8t_s45000.pt'
if ck.exists():
    G.productize(ck, 'TinyV8t', 2, '0.87M 256域 w_gan=0.25 45k步 (同速加强对抗)')
else:
    log('!! v8t 最终 ckpt 不存在, 跳过打包', LOG)

if v9b is not None and v9b >= 0.55:
    log('[branch] v9b=%s >= 0.55 => 512 域 + 容量 (0.87M, 12k 步)' % v9b, LOG)
    run('train v10a (512域 0.87M 12k)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '12000',
                                         '--bs', '4', '--res', '512', '--ch', '32,32,32,64',
                                         '--tag', 'v10a'], timeout=43200)
    ck = CK / 'v10a_s12000.pt'
    if ck.exists():
        G.productize(ck, 'TinyV10a', 1, '0.87M 512域 down=1 12k步 (分辨率+容量)')
    else:
        log('!! v10a 最终 ckpt 不存在, 跳过打包', LOG)
else:
    log('[branch] v9b=%s < 0.55 => 256 域加宽 (3.5M, 20k 步)' % v9b, LOG)
    run('train v8w (256域 3.5M 20k)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '20000',
                                       '--bs', '4', '--res', '256', '--ch', '64,64,64,128',
                                       '--tag', 'v8w'], timeout=43200)
    ck = CK / 'v8w_s20000.pt'
    if ck.exists():
        G.productize(ck, 'TinyV8w', 2, '3.5M 256域 20k步 (容量)')
    else:
        log('!! v8w 最终 ckpt 不存在, 跳过打包', LOG)
log('CHAIN_G_DONE', LOG)
