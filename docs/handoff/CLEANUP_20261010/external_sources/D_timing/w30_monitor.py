# -*- coding: utf-8 -*-
"""P100 1.4B w30/w31 训练监控 (2026-09-04 v2, 修复 ls glob bug):
监控 w30/w31 新训 (2500 步) 至完成 (进程消失 + 双双 lora_s2500.pt + [done])。
崩溃/超时(7h) -> CRITICAL 退出交人工。完成后退出 (决策由主会话做, autoval 已自动门控)。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, r'D:/tmp_timing')
from p100_upload import connect

LOG = Path(r'D:/tmp_timing/w30_monitor.log')
VARIANTS = ['p512_14b_w30', 'p512_14b_w31']
TARGET_CKPT = 'lora_s2500.pt'


def log(m):
    line = time.strftime('[%m-%d %H:%M:%S] ') + m
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def remote(cmd, timeout=60):
    cli = connect()
    try:
        _i, _o, _e = cli.exec_command(cmd, timeout=timeout)
        out = _o.read().decode('utf-8', 'replace')
        err = _e.read().decode('utf-8', 'replace')
        return out + (('[stderr] ' + err) if err.strip() else '')
    finally:
        cli.close()


def main():
    log('[boot] w30/w31 monitor: 盯到 2500 步完成')
    stable = 0
    t_start = time.time()
    while True:
        if time.time() - t_start > 7 * 3600:
            log('[CRITICAL] 超过 7h 未完成, 人工介入')
            sys.exit(2)
        try:
            r = remote("pgrep -f 'train_vosr_lora_14[b]' | wc -l; "
                       "ls ~/p100/runs/p512_14b_w30/lora_s*.pt 2>/dev/null | tail -1; "
                       "ls ~/p100/runs/p512_14b_w31/lora_s*.pt 2>/dev/null | tail -1; "
                       "tail -1 ~/p100/logs/p512_14b_w30.log; "
                       "tail -1 ~/p100/logs/p512_14b_w31.log", timeout=60)
        except Exception as e:
            log(f'[warn] poll 失败: {e}')
            time.sleep(120)
            continue
        lines = [l for l in r.splitlines() if l.strip()]
        nproc = int(lines[0]) if lines else -1
        ck30 = lines[1] if len(lines) > 1 else '?'
        ck31 = lines[2] if len(lines) > 2 else '?'
        log(f'[poll] procs={nproc} w30_newest={ck30} w31_newest={ck31}')
        if nproc > 0:
            stable = 0
            time.sleep(180)
            continue
        ok30 = ck30 == TARGET_CKPT
        ok31 = ck31 == TARGET_CKPT
        if ok30 and ok31:
            stable += 1
            if stable >= 2:
                log('[complete] w30/w31 均到 2500, 训练完成 — 等主会话决策')
                sys.exit(0)
        else:
            log(f'[CRITICAL] w30/w31 进程消失但未完成: w30={ck30} w31={ck31} — 疑似崩溃, 人工介入')
            sys.exit(1)
        time.sleep(180)


if __name__ == '__main__':
    main()
