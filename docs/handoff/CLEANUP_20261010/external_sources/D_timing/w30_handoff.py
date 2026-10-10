# -*- coding: utf-8 -*-
"""P100 1.4B w30 handoff watcher (2026-09-04):
等 w20/w15 续训完成 (进程消失 + 双双 lora_s6000.pt) -> 自动启动 w30/w31 新训 ->
验证启动后退出。崩溃/超时则 CRITICAL 退出交人工。
前置: p100_autoval_14b.py 的 VARIANTS 已含 w30/w31 (链已重启)。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, r'D:/tmp_timing')
from p100_upload import connect

LOG = Path(r'D:/tmp_timing/w30_handoff.log')


def log(m):
    line = time.strftime('[%m-%d %H:%M:%S] ') + m
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def remote(cmd, timeout=120):
    cli = connect()
    try:
        _i, _o, _e = cli.exec_command(cmd, timeout=timeout)
        out = _o.read().decode('utf-8', 'replace')
        err = _e.read().decode('utf-8', 'replace')
        return out + (('[stderr] ' + err) if err.strip() else '')
    finally:
        cli.close()


def main():
    log('[boot] w30 handoff watcher: 等 w20/w15 续训到 6000 后启动 w30/w31')
    stable = 0
    t_start = time.time()
    while True:
        if time.time() - t_start > 7 * 3600:
            log('[CRITICAL] 超过 7h 未完成, 人工介入')
            sys.exit(2)
        try:
            # 注意: pgrep 模式用 14[b] 防自匹配
            r = remote("pgrep -f 'train_vosr_lora_14[b]' | wc -l; "
                       "ls ~/p100/runs/p512_14b_w20/ 2>/dev/null | tail -1; "
                       "ls ~/p100/runs/p512_14b_w15/ 2>/dev/null | tail -1; "
                       "tail -1 ~/p100/logs/p512_14b_w20_resume.log; "
                       "tail -1 ~/p100/logs/p512_14b_w15_resume.log", timeout=60)
        except Exception as e:
            log(f'[warn] poll 失败: {e}')
            time.sleep(120)
            continue
        lines = [l for l in r.splitlines() if l.strip()]
        nproc = int(lines[0]) if lines else -1
        ck20 = lines[1] if len(lines) > 1 else '?'
        ck15 = lines[2] if len(lines) > 2 else '?'
        log(f'[poll] procs={nproc} w20_newest={ck20} w15_newest={ck15}')
        if nproc > 0:
            stable = 0
            time.sleep(180)
            continue
        ok20 = 's6000' in ck20
        ok15 = 's6000' in ck15
        if ok20 and ok15:
            stable += 1
            if stable >= 2:
                log('[complete] w20/w15 均到 6000, 启动 w30/w31 新训')
                out = remote('bash ~/p100/tools/p100_w30_launch.sh 2>&1', timeout=600)
                log('--- launch output ---\n' + out)
                time.sleep(90)
                try:
                    r2 = remote("pgrep -f 'train_vosr_lora_14[b].*p512_14b_w3' | wc -l", timeout=60)
                    log(f'[verify] w3x 进程数: {r2.strip()}')
                    r3 = remote("grep -E 'boot' ~/p100/logs/p512_14b_w30.log ~/p100/logs/p512_14b_w31.log 2>/dev/null | head -4", timeout=60)
                    log('[verify] boot lines:\n' + r3)
                except Exception as e:
                    log(f'[warn] verify 失败: {e}')
                log('[done] handoff 完成 — w30/w31 训练中, autoval 将自动门控 s500-s2500')
                sys.exit(0)
        else:
            log(f'[CRITICAL] 训练进程消失但未到 6000: w20={ck20} w15={ck15} — 疑似崩溃, 人工介入')
            sys.exit(1)
        time.sleep(180)


if __name__ == '__main__':
    main()
