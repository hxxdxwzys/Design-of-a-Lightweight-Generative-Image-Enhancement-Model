# -*- coding: utf-8 -*-
"""2026-09-15 链 H2 (替换 H1 的退化 v11; 单进程串行)
背景: v11 的判别器在 512 域被秒杀 (gan≈1.0, D≈0.001) => 对抗项失效, 退化成纯 L1+LPIPS@512。
修法: --d_band 1 (判别器输入先 band-limit, 去掉"清晰度"这个平凡线索, 只判结构/伪影) —— 冒烟已验证 D≈0.25 恢复平衡。
H1': v11b = 256 域学生 + 512 域损失/GAN + band-limit D, 25k 步, --init_from v11_s6000.pt (保留 6k 步的 512 域重建学习)
H2 : v9b2 = 512 域 0.38M, 45k 步 (已验证配方加长)
H3 : 诊断窗口 = MUSIQ/MANIQA 体检 (官方无参四项里我们从没看过的两项) + 原生分辨率/噪声探针
日志: D:/tmp_timing/chain_20260915h2.log
"""
import json, subprocess, sys, time, shutil
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
import gan_chain as G

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'chain_20260915h2.log'
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


def wait_gpu_free(thresh=2200, need=3, ceiling_h=2):
    t0 = time.time(); ok = 0
    while time.time() - t0 < ceiling_h * 3600:
        u = gpu_used()
        ok = ok + 1 if 0 <= u < thresh else 0
        log('  [gpu] %s MiB (%d/%d)' % (u, ok, need))
        if ok >= need:
            return True
        time.sleep(45)
    return False


G.run = run
G.LOG = LOG
G.log = log

log('=== 链 H2 启动 ===')
if not wait_gpu_free():
    log('!! GPU 不空闲, 仍继续')

# ---------------- H1': v11b ----------------
init = CK / 'v11_s6000.pt'
run('train v11b (256域 + 512域损失 + band-limit D, 25k)',
    [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '25000', '--bs', '6', '--res', '256',
     '--loss_res', '512', '--d_band', '1', '--ch', '32,32,32,64', '--tag', 'v11b',
     '--init_from', str(init)], timeout=21600)
ck = CK / 'v11b_s25000.pt'
if ck.exists():
    G.productize(ck, 'TinyV11b', 2, '0.87M 256域 + 512域损失/GAN + bandlimitD')
else:
    log('!! v11b 最终 ckpt 缺失, 跳过打包')

# ---------------- H2: v9b2 ----------------
run('train v9b2 (512域 0.38M 45k)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '45000',
                                     '--bs', '4', '--res', '512', '--ch', '32,32,32',
                                     '--tag', 'v9b2'], timeout=43200)
ck = CK / 'v9b2_s45000.pt'
if ck.exists():
    G.productize(ck, 'TinyV9b2', 1, '0.38M 512域 45k步 (v9b 配方加长)')
else:
    log('!! v9b2 最终 ckpt 缺失, 跳过打包')

# ---------------- H3: 诊断窗口 ----------------
TESTLQ = 'C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/赛题二/测试集'
diag = [
    ('LQ基线', TESTLQ),
    ('M24', 'D:/models/final_compliant/output_dir'),
    ('TinyV7gan', 'D:/models/pkg_TinyV7gan/output_dir'),
    ('TinyV9b', 'D:/models/pkg_TinyV9b/output_dir'),
    ('TinyV9b2', 'D:/models/pkg_TinyV9b2/output_dir'),
    ('TinyV11b', 'D:/models/pkg_TinyV11b/output_dir'),
]
for tag, d in diag:
    if Path(d).exists():
        run('diag_musiq_' + tag, [PY, '-u', str(T / 'measure_dir.py'), '--dir', d, '--n', '30',
                                  '--metrics', 'musiq,maniqa,clipiqa,niqe', '--tag', 'diag_' + tag])
run('measure_probe_native30', [PY, '-u', str(T / 'measure_dir.py'), '--dir', 'D:/models/probe_native',
                               '--n', '30', '--metrics', 'niqe,clipiqa,lapvar', '--tag', 'probe_native30'])

log('=== 产物汇总 ===')
for f in sorted(Path('D:/models').glob('pkg_*/test30_metrics.json')):
    try:
        j = json.loads(f.read_text(encoding='utf-8'))
        log('%-18s clipiqa=%-8s niqe=%-8s n=%s' % (f.parent.name, j.get('clipiqa'), j.get('niqe'), j.get('n')))
    except Exception as e:
        log('!! %s %s' % (f, e))
for f in sorted(Path('D:/models').glob('metrics_diag_*.json')):
    try:
        j = json.loads(f.read_text(encoding='utf-8'))
        log('%-24s musiq=%-8s maniqa=%-8s clipiqa=%-8s niqe=%-8s' % (f.stem, j.get('musiq'), j.get('maniqa'), j.get('clipiqa'), j.get('niqe')))
    except Exception as e:
        log('!! %s %s' % (f, e))
log('CHAIN_H2_DONE')
