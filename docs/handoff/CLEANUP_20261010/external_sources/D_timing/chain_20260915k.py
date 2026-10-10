# -*- coding: utf-8 -*-
"""链 K (2026-09-15) — 接管 GPU: 等 v11b 收尾 -> 掐掉 v9b2 -> 跑无参面板 -> 测 1.4B 家族时延

决策依据 (2026-09-15 08:5x):
  v11b 让跑完 (25 域新机制, 还剩 ~1.7h); v9b2 (45k, 8.4h, 期望 <+1.5 分) 延后;
  chain J (4K 有参) 延后 —— 它自己的章程就写着不得复活保真轴。
  插入: 官方无参四项面板 (MUSIQ/MANIQA 从未测过) + 1.4B 家族 512 时延 (去掉"分数低是速度还是质量"的混淆)。
日志: D:/tmp_timing/chain_20260915k.log
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG = T / 'chain_20260915k.log'
H2LOG = T / 'chain_20260915h2.log'


def log(m):
    s = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(s, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


def run(name, cmd, timeout=10800):
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


def gpu_used():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=60)
        return int((p.stdout or '0').strip().splitlines()[0])
    except Exception:
        return -1


def kill_by_cmdline(patterns):
    """按命令行子串杀 python 进程 (不匹配自己)"""
    ps = ('Get-CimInstance Win32_Process -Filter "Name=\'python.exe\'" | '
          'Where-Object { $_.CommandLine -ne $null -and $_.CommandLine -notlike \'*chain_20260915k*\' '
          'and (%s) } | ForEach-Object { Write-Output ("kill " + $_.ProcessId + " " + $_.CommandLine.Substring(0,[Math]::Min(90,$_.CommandLine.Length))); '
          'Stop-Process -Id $_.ProcessId -Force }')
    cond = ' -or '.join('$_.CommandLine -like \'*%s*\'' % p for p in patterns)
    last = ''
    for shell in ('pwsh', 'powershell'):
        cmd = [shell, '-NoProfile', '-Command', ps % cond]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                               errors='replace', timeout=120)
            out = (p.stdout or '') + (p.stderr or '')
            if p.returncode == 0:
                return out
            last = '%s rc=%s %s' % (shell, p.returncode, out[:200])
        except Exception as e:
            last = '%s EXC %s' % (shell, e)
    return 'KILL_FAILED ' + last


log('=== 链 K 启动: 等 v11b 收尾 ===')
t0 = time.time()
seen_v11b = (Path('D:/models/tiny_v1/v11b_s25000.pt')).exists()
while time.time() - t0 < 4 * 3600:
    txt = H2LOG.read_text(encoding='utf-8', errors='replace') if H2LOG.exists() else ''
    if 'train v9b2' in txt:
        log('检测到 H2 已进入 v9b2, 掐断')
        break
    if 'CHAIN_H2_DONE' in txt:
        log('H2 自然结束')
        break
    time.sleep(20)
else:
    log('!! 等待超时, 仍继续')

log('kill 输出:\n' + kill_by_cmdline(['chain_20260915h2.py', '--tag v9b2']))
time.sleep(15)
log('  [gpu] %s MiB' % gpu_used())

v11 = Path('D:/models/tiny_v1/v11b_s25000.pt')
log('v11b_s25000 存在=%s' % v11.exists())
pkg = Path('D:/models/pkg_TinyV11b/output_dir')
log('pkg_TinyV11b 出图数=%d' % len(list(pkg.glob('*.jpg'))) if pkg.exists() else 'pkg_TinyV11b 不存在')

# 1.4B 家族 512 时延 (去掉"1.4B 低分是速度还是质量"的混淆)
for tag, md in [('w2_m0s1_s090', 'D:/models/wave2_m0s1_s090/model_dir'),
                ('w2_m0s7_s097', 'D:/models/wave2_m0s7_s097/pack_model_dir'),
                ('w2_m0s1_s097', 'D:/models/wave2_m0s1_s097/model_dir'),
                ('w2_m0s1_s100', 'D:/models/wave2_m0s1_s100/model_dir')]:
    if Path(md).exists():
        run('latency ' + tag, [PY, '-u', str(T / 'measure_latency.py'), md], timeout=3600)

# 无参四项面板 v2: 多口径 x 多指标 (native 跳过 musiq: 109s/张)
run('panel n=10 multi-mode', [PY, '-u', str(T / 'diag_panel.py'), '--n', '10',
                              '--modes', 'ms512,ms1024,c1024,native',
                              '--metrics', 'clipiqa,musiq,maniqa,niqe',
                              '--native-skip', 'musiq'], timeout=14400)

log('CHAIN_K_DONE')
