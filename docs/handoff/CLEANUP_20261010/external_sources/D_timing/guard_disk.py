# -*- coding: utf-8 -*-
"""w48 训练磁盘守护: 每 --interval 秒检查 D: 空闲, 不足 --free-gb 时按优先级删旧 ckpt:
  1) full_s{5k,10k,15k} (resume 用不上, 只留 full_final)
  2) net_s<10000 (最旧先删)
  3) ema_s<10000 (最旧先删, 最不优先——ema 便宜且可能有逐图价值)
始终保留: net/ema s>=10000 (供直选 sweep)、full_final.pt、eval_log.json。
用法 (脱离会话):
  Start-Process D:/Python/python.exe -ArgumentList 'D:/tmp_timing/guard_disk.py','--out','D:/models/metric_net_w48' -WindowStyle Hidden
"""
import argparse
import re
import shutil
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--interval', type=int, default=600)
    ap.add_argument('--free-gb', type=float, default=12.0)
    ap.add_argument('--keep-early', action='store_true',
                    help='保留 net/ema s<10000 (供增补 sweep), 仅删 full_s5k/10k/15k')
    args = ap.parse_args()

    out = Path(args.out)
    while True:
        free_gb = shutil.disk_usage(out).free / 1e9
        if free_gb < args.free_gb:
            cands = []
            for p in out.glob('*.pt'):
                m = re.match(r'(net_s|ema_s|full_s)(\d+)\.pt$', p.name)
                if not m:
                    continue
                kind, step = m.group(1), int(m.group(2))
                prio = None
                if kind == 'full_s' and step < 100000:
                    prio = 0
                elif kind == 'net_s' and step < 10000 and not args.keep_early:
                    prio = 1
                elif kind == 'ema_s' and step < 10000 and not args.keep_early:
                    prio = 2
                if prio is not None:
                    cands.append((prio, step, p))
            cands.sort()
            if not cands:
                print(f'[guard] free={free_gb:.1f}GB < {args.free_gb} 但无可删 ckpt!', flush=True)
            for prio, step, p in cands:
                if shutil.disk_usage(out).free / 1e9 >= args.free_gb + 1.0:
                    break
                p.unlink()
                print(f'[guard] free={shutil.disk_usage(out).free / 1e9:.1f}GB 删除 {p.name}', flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
