# -*- coding: utf-8 -*-
"""链 L (2026-09-15 下午) — 时延复测 -> v11b 轨迹 -> v12(续训) -> v13(3.5M 宽) -> v9b2(填充)

依据 prereg_20260915b.txt:
  v11b 0.5481@57.5x => 预测 46.1 < v9b 的 47.8 => 不提交 (已机械判分)
  官方口径 = 保留细节类(native/c1024), 非 resize512 => native test30 clipiqa 仍是正确代理
  1.4B 家族无隐藏轴(四项全差于 M24, 真实时延 0.98 s 而非 12.64 s)
  唯一活着的合规杠杆: 在高速档把 native clipiqa 往上推
日志: D:/tmp_timing/chain_20260915l.log
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
import gan_chain as G

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
CK = Path('D:/models/tiny_v1')
LOG = T / 'chain_20260915l.log'


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

log('=== 链 L 启动 ===')

# 1) 时延复测 (修好 CPU 输入 bug 之后)
for tag in ['TinyV7gan', 'TinyV9b', 'TinyV11b']:
    md = Path('D:/models/pkg_%s/model_dir' % tag)
    if md.exists():
        run('latency ' + tag, [PY, '-u', str(T / 'measure_latency.py'), str(md)], timeout=1800)

# 2) v11b 训练轨迹 -> 决定"继续训练"是否还有斜率
cqs = {}
for step in (8000, 16000, 24000):
    ck = CK / ('v11b_s%d.pt' % step)
    if not ck.exists():
        log('!! %s 缺失' % ck.name); continue
    od = Path('D:/models/traj_v11b/s%d' % step)
    run('render v11b_s%d' % step, [PY, '-u', str(T / 'render_subset.py'), '--ckpt', str(ck),
                                   '--out', str(od), '--down', '2', '--n', '30'], timeout=3600)
    run('measure v11b_s%d' % step, [PY, '-u', str(T / 'measure_dir.py'), '--dir', str(od),
                                    '--n', '30', '--metrics', 'clipiqa,niqe,lapvar',
                                    '--tag', 'v11b_s%d' % step], timeout=3600)
    jf = Path('D:/models/traj_v11b') / ('metrics_v11b_s%d.json' % step)
    if jf.exists():
        try:
            cqs[step] = float(json.loads(jf.read_text(encoding='utf-8'))['clipiqa'])
        except Exception as e:
            log('!! 读 %s 失败 %s' % (jf, e))
cqs[25000] = 0.5481   # 已知(打包时测得)
log('[traj] v11b clipiqa: ' + ' '.join('%s=%.4f' % (k, v) for k, v in sorted(cqs.items())))
slope = None
if 16000 in cqs and 24000 in cqs:
    slope = (cqs[24000] - cqs[16000]) / 8.0   # 每 1000 步
    log('[traj] s16000->s24000 斜率 = %+.5f / 1000 步' % slope)

# 3) v12 = 续训 (仅当轨迹还有斜率)
do_v12 = slope is not None and slope > 0.00125
log('[decide] 续训 v12 = %s (阈值 +0.00125/1000步 = +0.01/8k)' % do_v12)
if do_v12:
    run('train v12 (续 v11b 25k)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '25000',
                                    '--bs', '6', '--res', '256', '--loss_res', '512', '--d_band', '1',
                                    '--ch', '32,32,32,64', '--tag', 'v12',
                                    '--init_from', str(CK / 'v11b_s25000.pt')], timeout=21600)
    if (CK / 'v12_s25000.pt').exists():
        G.productize(CK / 'v12_s25000.pt', 'TinyV12', 2, '续 v11b 25k (512域损失+bandlimitD)')

# 4) v13 = 3.5M 宽 + 同配方 (从未跑过的 §一D 步骤①)
init = CK / ('v12_s25000.pt' if (CK / 'v12_s25000.pt').exists() else 'v11b_s25000.pt')
run('train v13 (3.5M 宽 + 512域损失 + bandlimitD)', [PY, '-u', str(T / 'train_tiny_gan.py'),
    '--steps', '20000', '--bs', '4', '--res', '256', '--loss_res', '512', '--d_band', '1',
    '--ch', '64,64,64,128', '--tag', 'v13', '--init_from', str(init)], timeout=28800)
if (CK / 'v13_s20000.pt').exists():
    G.productize(CK / 'v13_s20000.pt', 'TinyV13', 2, '3.5M 宽 + 512域损失 + bandlimitD')

# 5) v9b2 = 512 域加长 (填充)
run('train v9b2 (512域 0.38M 45k)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '45000',
                                     '--bs', '4', '--res', '512', '--ch', '32,32,32', '--tag', 'v9b2'], timeout=43200)
if (CK / 'v9b2_s45000.pt').exists():
    G.productize(CK / 'v9b2_s45000.pt', 'TinyV9b2', 1, '0.38M 512域 45k (v9b 配方加长)')

log('=== 产物汇总 ===')
for f in sorted(Path('D:/models').glob('pkg_*/test30_metrics.json')):
    try:
        j = json.loads(f.read_text(encoding='utf-8'))
        log('%-18s clipiqa=%-8s niqe=%-8s n=%s' % (f.parent.name, j.get('clipiqa'), j.get('niqe'), j.get('n')))
    except Exception as e:
        log('!! %s %s' % (f, e))
log('CHAIN_L_DONE')
