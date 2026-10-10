# -*- coding: utf-8 -*-
"""1.4B 全档门控完成监视器 (2026-09-04): 等 w15 尾档(6) + w30(4) + w31(5) 共 15 个门控落地。
判定: gate_p512_14b_w31_lora_s2500.json 存在 (最后一个)。超时 3h 告警退出。
"""
import sys
import time
from pathlib import Path

OUT = Path(r'D:/tmp_timing/p100_val')
LOG = Path(r'D:/tmp_timing/gates_done_monitor.log')

EXPECTED = [
    'p512_14b_w15_lora_s3500', 'p512_14b_w15_lora_s4000', 'p512_14b_w15_lora_s4500',
    'p512_14b_w15_lora_s5000', 'p512_14b_w15_lora_s5500', 'p512_14b_w15_lora_s6000',
    'p512_14b_w30_lora_s1000', 'p512_14b_w30_lora_s1500', 'p512_14b_w30_lora_s2000', 'p512_14b_w30_lora_s2500',
    'p512_14b_w31_lora_s500', 'p512_14b_w31_lora_s1000', 'p512_14b_w31_lora_s1500',
    'p512_14b_w31_lora_s2000', 'p512_14b_w31_lora_s2500',
]


def log(m):
    line = time.strftime('[%m-%d %H:%M:%S] ') + m
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    log('[boot] 等待 15 个 1.4B 尾档门控完成')
    t0 = time.time()
    while True:
        if time.time() - t0 > 3 * 3600:
            log('[CRITICAL] 3h 超时, 人工介入')
            sys.exit(2)
        got = [e for e in EXPECTED if (OUT / f'gate_{e}.json').exists()]
        if len(got) >= len(EXPECTED):
            log(f'[done] 15/15 门控齐 — 等主会话终判')
            sys.exit(0)
        # 心跳式输出每 10 轮
        if int(time.time()) % 600 < 120:
            log(f'[progress] {len(got)}/15: {got[-1] if got else "-"}')
        time.sleep(120)


if __name__ == '__main__':
    main()
