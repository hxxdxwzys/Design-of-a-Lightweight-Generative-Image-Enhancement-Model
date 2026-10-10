# -*- coding: utf-8 -*-
"""链 F (加固版): 等 E2 结束 -> 独占 GPU 测量窗口"""
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
    """关键加固: subprocess.run 超时/异常一律不抛出, 只记录 -> 队列永不因单步失败而崩"""
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

LOG = T / 'chain_20260915f.log'
E2LOG = T / 'chain_20260915e2.log'
TRAINLOG = 'D:/models/tiny_v1/train_gan.log'


def run(name, cmd, timeout=10800):
    log('>>> ' + name, LOG)
    t0 = time.time()
    rc, out = safe_run(name, cmd, timeout)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(out + '\n')
    log('<<< %s rc=%d (%.1f min)' % (name, rc, (time.time() - t0) / 60), LOG)
    return rc, out


log('等待夜间链 E2 (CHAIN_E2_DONE) ...', LOG)
t0 = time.time()
while time.time() - t0 < 14 * 3600:
    txt = E2LOG.read_text(encoding='utf-8', errors='replace') if E2LOG.exists() else ''
    if 'CHAIN_E2_DONE' in txt:
        break
    if time.time() - t0 > 1800 and stale(TRAINLOG, 1500) and stale(str(E2LOG), 1800):
        log('!! E2 疑似中断 (训练日志与链日志均 25 分钟未更新), 继续测量窗口', LOG)
        break
    time.sleep(120)
log('进入测量窗口 (等待 %.1f h)' % ((time.time() - t0) / 3600), LOG)

if wait_gpu_free(LOG, thresh=4000):
    run('measure_probe_native30', [PY, '-u', str(T / 'measure_dir.py'), '--dir', 'D:/models/probe_native',
                                   '--n', '30', '--metrics', 'niqe,clipiqa,lapvar', '--tag', 'probe_native30'])
    import shutil
    for s in ('s0.050', 's0.100'):
        d = Path('D:/models/probe_noise') / s
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    run('render_noise_005_010', [PY, '-u', str(T / 'probe_noise.py'),
                                 '--ckpt', 'D:/models/tiny_v1/v7gan_s25000.pt',
                                 '--out', 'D:/models/probe_noise', '--sigmas', '0.05,0.10', '--n', '10'])
    for s in ('s0.000', 's0.050', 's0.100'):
        d = 'D:/models/probe_noise/' + s
        if Path(d).exists():
            run('measure_noise_' + s, [PY, '-u', str(T / 'measure_dir.py'), '--dir', d, '--n', '10',
                                       '--metrics', 'niqe,clipiqa,lapvar', '--tag', 'noise' + s.replace('.', '')])
    # ---- 恢复: E2 若在打包前崩掉, 用最新 ckpt 补打包 (不浪费几小时训练) ----
    CK = Path('D:/models/tiny_v1')
    for tag, down, pkg in (('v9b', 1, 'TinyV9b'), ('v9a', 1, 'TinyV9a'), ('v8s', 2, 'TinyV8s')):
        mj = Path('D:/models') / ('pkg_' + pkg) / 'test30_metrics.json'
        if mj.exists():
            continue
        cks = sorted(CK.glob(tag + '_s*.pt'), key=lambda p: int(p.stem.split('_s')[-1]))
        if not cks:
            continue
        log('!! %s 缺打包, 用 %s 恢复' % (pkg, cks[-1].name), LOG)
        G.LOG = LOG
        G.log = lambda m: log(m, LOG)
        try:
            G.productize(cks[-1], pkg, down, 'E2 中断后的恢复打包 (%s)' % cks[-1].name)
        except Exception as e:
            log('!! 恢复打包失败 %s: %s' % (pkg, e), LOG)

    log('=== 汇总 (标准口径) ===', LOG)
    for f2 in sorted(Path('D:/models').rglob('metrics_*.json')):
        try:
            j = json.loads(f2.read_text(encoding='utf-8'))
            log('%-32s clipiqa=%-8s niqe=%-8s lapvar=%-9s n=%s' % (f2.stem, j.get('clipiqa'), j.get('niqe'), j.get('lapvar'), j.get('n')), LOG)
        except Exception as e:
            log('!! %s %s' % (f2, e), LOG)
else:
    log('!! GPU 一直不空闲, 跳过测量窗口', LOG)
log('CHAIN_F_DONE', LOG)
