# -*- coding: utf-8 -*-
"""等待 w_lpips 轴全档门控完成 (2026-09-09 20:35)
每 5 分钟轮询 p100_val/gate_p512_14b_m0s1wl{2,3}_lora_s*.json; 24 档齐 -> 打印判定表并退出。
判定: 预登记过门 = clip>=0.920 且 lpips<=0.352 (vs 冠军 m0s1-sc0.93 = 0.9261/0.3571)
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

VAL = Path(r'D:/tmp_timing/p100_val')
STEPS = list(range(500, 6001, 500))
TAGS = [(w, s) for w in (2, 3) for s in STEPS]
CHAMP = (0.9261, 0.3571)
POLL, TIMEOUT = 300, 8 * 3600


def gate(w, s):
    return VAL / f'gate_p512_14b_m0s1wl{w}_lora_s{s}.json'


def main():
    t0 = time.time()
    while True:
        have = [(w, s) for w, s in TAGS if gate(w, s).exists()]
        print(f'[{time.strftime("%m-%d %H:%M:%S")}] 门控进度 {len(have)}/{len(TAGS)}', flush=True)
        if len(have) == len(TAGS):
            break
        if time.time() - t0 > TIMEOUT:
            print('[超时退出]', flush=True)
            break
        time.sleep(POLL)

    print('\n=== w_lpips 轴全档门控结果 ===', flush=True)
    for w in (2, 3):
        print(f'\n-- w_lpips={w} (w_clip 2.0, r32, seed1) --')
        best = None
        for s in STEPS:
            p = gate(w, s)
            if not p.exists():
                print(f'  s{s:<5} MISSING')
                continue
            g = json.loads(p.read_text(encoding='utf-8'))
            flag = ''
            if g['clip'] >= 0.920 and g['lpips'] <= 0.352:
                flag = '  ⭐ 过预登记门!'
            if g['clip'] >= 0.920:
                if best is None or g['lpips'] < best[1]:
                    best = (s, g['lpips'], g['clip'])
            print(f'  s{s:<5} clip={g["clip"]:.4f} lpips={g["lpips"]:.4f} '
                  f'pass={g["pass"]}{flag}')
        if best:
            print(f'  → clip≥0.920 内最优: s{best[0]} ({best[2]:.4f}, {best[1]:.4f}) '
                  f'vs 冠军 (0.9261, 0.3571) = {best[1]-CHAMP[1]:+.4f}')
        else:
            print('  → 无 clip≥0.920 的点')
    print('\n判定: 若最优 lpips > 0.352 → 与 rank/多阶段一致 = 前沿不可下移 (第四条证据)', flush=True)


if __name__ == '__main__':
    main()
