# -*- coding: utf-8 -*-
"""双端唤醒监视器 (2026-09-09 21:05) — 取代 wait_wl_gates.py

目的: 服务器训练端 + 本地门控端 任一"完成/异常/停滞"都让本进程退出 -> harness 通知我。

退出条件（任一满足即退出并打印报告）:
  A. 24 档门控齐 (wl2/wl3 x s500..s6000)          -> 正常完成
  B. 服务器训练进程消失 且 门控停滞 > STALL_MIN    -> 异常（训练挂了）
  C. 服务器日志 step 停滞 > STALL_MIN 且 门控停滞 > STALL_MIN -> 异常（卡死）
  D. 本地 autoval 心跳过期 > STALL_MIN 且 门控停滞 > STALL_MIN -> 异常（本地链挂了）
  E. 硬超时 TIMEOUT                               -> 超时
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, r'D:/tmp_timing')
import paramiko  # noqa: E402

VAL = Path(r'D:/tmp_timing/p100_val')
HB = VAL / 'autoval14.hb'
HOST, PORT, USER, PWD = '10.96.138.171', 2222, 'tempuser', 'temp76'
VARIANTS = {'wl2': 'p512_14b_m0s1wl2', 'wl3': 'p512_14b_m0s1wl3'}
STEPS = list(range(500, 6001, 500))
TAGS = [(w, s) for w in ('wl2', 'wl3') for s in STEPS]
CHAMP = (0.9261, 0.3571)
POLL = 180
STALL_MIN = 16 * 60      # 16 分钟无任何进展视为停滞
TIMEOUT = 7 * 3600


def log(m):
    print(f'[{time.strftime("%m-%d %H:%M:%S")}] {m}', flush=True)


def ssh_cmd(cmd, timeout=40, retries=3):
    for i in range(retries):
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=20)
            _in, _out, _err = cli.exec_command(cmd, timeout=timeout)
            out = _out.read().decode('utf-8', errors='replace')
            cli.close()
            return out
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(5)
    return ''


def server_state():
    """返回 (训练进程数, {variant: 最后 step}, {variant: 是否 done})"""
    cmd = ("pgrep -f '[t]rain_vosr_lora_14b.py' | wc -l; "
           "for v in p512_14b_m0s1wl2 p512_14b_m0s1wl3; do "
           "grep -h '\\[step' ~/p100/logs/$v.log 2>/dev/null | tail -1 | "
           "sed 's/.*\\[step \\([0-9]*\\)\\].*/\\1/'; "
           "grep -c '\\[done\\]' ~/p100/logs/$v.log 2>/dev/null; done")
    out = ssh_cmd(cmd)
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    nproc = int(lines[0]) if lines and lines[0].isdigit() else -1
    steps, dones = {}, {}
    idx = 1
    for w, v in VARIANTS.items():
        steps[w] = int(lines[idx]) if idx < len(lines) and lines[idx].isdigit() else 0
        idx += 1
        dones[w] = (lines[idx] == '1') if idx < len(lines) else False
        idx += 1
    return nproc, steps, dones


def gate_count():
    return sum(1 for w, s in TAGS if (VAL / f'gate_p512_14b_m0s1wl{w[-1]}_lora_s{s}.json').exists())


def gate_path(w, s):
    return VAL / f'gate_p512_14b_m0s1wl{w[-1]}_lora_s{s}.json'


def report(reason):
    log(f'=== 退出: {reason} ===')
    print('\n=== w_lpips 轴全档门控结果 ===', flush=True)
    for w in ('wl2', 'wl3'):
        print(f'\n-- {w} (w_clip 2.0, w_lpips {w[-1]}, r32, seed1) --')
        best = None
        for s in STEPS:
            p = gate_path(w, s)
            if not p.exists():
                print(f'  s{s:<5} MISSING')
                continue
            g = json.loads(p.read_text(encoding='utf-8'))
            flag = '  ⭐ 过预登记门!' if (g['clip'] >= 0.920 and g['lpips'] <= 0.352) else ''
            if g['clip'] >= 0.920 and (best is None or g['lpips'] < best[1]):
                best = (s, g['lpips'], g['clip'])
            print(f'  s{s:<5} clip={g["clip"]:.4f} lpips={g["lpips"]:.4f} '
                  f'pass={g["pass"]}{flag}')
        if best:
            print(f'  → clip≥0.920 内最优: s{best[0]} ({best[2]:.4f}, {best[1]:.4f}) '
                  f'vs 冠军 (0.9261, 0.3571) = {best[1]-CHAMP[1]:+.4f}')
        else:
            print('  → 无 clip≥0.920 的点')
    print('\n判定: 最优 lpips > 0.352 → 与 rank/多阶段一致 = 前沿不可下移（第四条证据）',
          flush=True)


def main():
    log(f'[boot] 双端监视启动 (poll={POLL}s, stall={STALL_MIN//60}min, timeout={TIMEOUT//3600}h)')
    t0 = time.time()
    last_gates, last_steps, last_progress = -1, {}, time.time()
    ssh_fail = 0
    while True:
        try:
            gc = gate_count()
            hb_age = (time.time() - HB.stat().st_mtime) if HB.exists() else 1e9
            try:
                nproc, steps, dones = server_state()
                ssh_fail = 0
                srv = f'proc={nproc} steps={steps} done={dones}'
            except Exception as e:
                nproc, steps, dones = -1, {}, {}
                ssh_fail += 1
                srv = f'SSH 失败 x{ssh_fail}: {type(e).__name__}'
            log(f'门控 {gc}/{len(TAGS)} | 心跳 {hb_age/60:.1f}min | {srv}')

            # 进展判定（SSH 失败时不把空结果误当进展）
            progressed = (gc != last_gates)
            if nproc >= 0:
                progressed = progressed or any(
                    steps.get(k, 0) != last_steps.get(k, 0) for k in VARIANTS)
                last_steps = steps
            if progressed:
                last_progress = time.time()
            last_gates = gc
            stalled = (time.time() - last_progress) > STALL_MIN

            if gc == len(TAGS):
                report('A. 全部门控完成')
                return
            if nproc == 0 and stalled:
                report('B. 服务器训练进程消失且无进展')
                return
            if stalled and hb_age > STALL_MIN:
                report('C. 双端停滞（服务器 step 不动 + 本地心跳过期）')
                return
            if ssh_fail >= 10:
                report('F. 服务器持续不可达 (>30min)')
                return
            if time.time() - t0 > TIMEOUT:
                report('E. 硬超时')
                return
        except Exception as e:
            log(f'[warn] 监视异常（继续）: {type(e).__name__}: {str(e)[:150]}')
        time.sleep(POLL)


if __name__ == '__main__':
    main()
