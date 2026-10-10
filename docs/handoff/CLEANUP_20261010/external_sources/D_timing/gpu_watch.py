# -*- coding: utf-8 -*-
"""GPU 看门狗 (2026-08-19): 以 GPU 占用为唤醒信号
逻辑: 先观察到"持续繁忙"(util>=30% 连续 BUSY_POLLS 次) 武装, 之后
      连续 IDLE_POLLS 次空闲(util<20%, 每次间隔 60s) -> 退出 0 (通知宿主 = 唤醒代理)
      超过 MAX_WAIT 小时 -> 退出 1 (超时, 防任务挂死永不触发)
用法: D:/Python/python.exe gpu_watch.py [--busy-polls 3] [--idle-polls 10] [--max-wait 30] [--tag "b_final"]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

LOG = Path(r'D:/tmp_timing/gpu_watch.log')
SUM = Path(r'D:/tmp_timing/gpu_watch_summary.json')


def gpu_util():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=20)
        line = out.stdout.strip().splitlines()[0]
        util, mem = [int(x.strip()) for x in line.split(',')]
        return util, mem
    except Exception:
        return -1, -1


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--busy-polls', type=int, default=3, help='武装前需连续繁忙次数')
    ap.add_argument('--idle-polls', type=int, default=10, help='连续空闲次数即触发 (10x60s=10min)')
    ap.add_argument('--max-wait', type=int, default=30, help='最长等待小时, 超时退出 1')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    log(f'[boot] gpu_watch tag={args.tag} busy_polls={args.busy_polls} idle_polls={args.idle_polls}')

    busy_streak = 0
    idle_streak = 0
    armed = False
    t0 = time.time()
    busy_since = None
    idle_since = None
    samples = []

    while True:
        util, mem = gpu_util()
        samples.append((time.strftime('%H:%M:%S'), util, mem))
        # 信号: 显存 (util 波动大不可靠; 任务运行 = 模型驻留 >=2GB, 空闲基线 <1.5GB)
        busy = mem >= 2000
        idle_now = mem < 1500
        if busy:
            busy_streak += 1
            idle_streak = 0
            if busy_streak >= args.busy_polls and not armed:
                armed = True
                busy_since = time.strftime('%H:%M:%S')
                log(f'[armed] GPU 持续繁忙 (mem={mem}MB x{busy_streak})')
        elif idle_now:
            busy_streak = 0
            if armed:
                idle_streak += 1
                if idle_streak == 1:
                    idle_since = time.strftime('%H:%M:%S')
                    log(f'[idle-start] GPU 转空闲 (mem={mem}MB)')
                if idle_streak >= args.idle_polls:
                    log(f'[trigger] GPU 已空闲 {idle_streak} 分钟 (busy {busy_since} -> idle {idle_since})')
                    SUM.write_text(json.dumps({
                        'tag': args.tag, 'armed': armed, 'trigger': 'idle',
                        'busy_since': busy_since, 'idle_since': idle_since,
                        'samples': samples[-40:],
                    }, indent=1), encoding='utf-8')
                    sys.exit(0)
        else:
            busy_streak = 0
            idle_streak = 0
        if (time.time() - t0) > args.max_wait * 3600:
            log(f'[timeout] 等待超时 {args.max_wait}h (可能任务挂死)')
            SUM.write_text(json.dumps({'tag': args.tag, 'trigger': 'timeout',
                                       'samples': samples[-40:]}, indent=1), encoding='utf-8')
            sys.exit(1)
        time.sleep(60)


if __name__ == '__main__':
    main()
