# -*- coding: utf-8 -*-
"""世代判定助手 (2026-09-05): 读 m0s* gate jsons → 汇总表 + 三态标记
用法: python gen_decide.py [variant前缀, 默认 p512_14b_m0s]
输出: 每 variant 最优饱和点 (clip>=0.920) 与最优 lpips 点 + 三态判定建议
"""
import json
import sys
from pathlib import Path

OUT = Path(r'D:/tmp_timing/p100_val')
PREFIX = sys.argv[1] if len(sys.argv) > 1 else 'p512_14b_m0s'
SAFE_CLIP = 0.920
TARGET_LPIPS = 0.3654  # m0s0-s4500 当前冠军


def main():
    files = sorted(OUT.glob(f'gate_{PREFIX}*.json'))
    variants = {}
    for f in files:
        v = f.name[len('gate_'):].rsplit('_lora_s', 1)[0]
        step = f.name.rsplit('_lora_s', 1)[1].replace('.json', '')
        g = json.loads(f.read_text(encoding='utf-8'))
        variants.setdefault(v, []).append((int(step), g['clip'], g['lpips']))
    print(f'=== 世代判定: {PREFIX} ({len(variants)} variants) ===')
    print(f'{"variant":<18}{"最佳饱和点(clip>=0.920)":<32}{"最低lpips点":<28}')
    best_sat = []  # (variant, step, clip, lpips)
    for v in sorted(variants):
        pts = sorted(variants[v])
        sat = [p for p in pts if p[1] >= SAFE_CLIP]
        best = min(sat, key=lambda p: p[2], default=None)
        minlp = min(pts, key=lambda p: p[2])
        if best:
            best_sat.append((v, best[0], best[1], best[2]))
            print(f'{v:<18}s{best[0]:<5}({best[1]:.4f},{best[2]:.4f}){"":<8}s{minlp[0]:<5}({minlp[1]:.4f},{minlp[2]:.4f})')
        else:
            print(f'{v:<18}无饱和点{"":<24}s{minlp[0]:<5}({minlp[1]:.4f},{minlp[2]:.4f})')
    print()
    if not best_sat:
        print('[三态] C: 全灭 — m0s0 为孤点, 世代2 = m0s0 续训 + w1.8 对照')
        return
    best_overall = min(best_sat, key=lambda x: x[3])
    beat_m0s0 = [x for x in best_sat if x[3] < TARGET_LPIPS - 1e-6]
    print(f'最优饱和点: {best_overall[0]} s{best_overall[1]} ({best_overall[2]:.4f}, {best_overall[3]:.4f})')
    print(f'vs 冠军 m0s0-s4500 (0.9258, {TARGET_LPIPS})')
    if beat_m0s0:
        for x in beat_m0s0:
            print(f'  > 超越冠军: {x[0]} s{x[1]} ({x[2]:.4f}, {x[3]:.4f})')
        print('[三态] A+: 再现且超越 — 世代2 = 种子扩展 + sc 微扫 + 前沿精扫')
    elif len(best_sat) >= 2:
        print('[三态] A: 再现 (多个饱和点) — 世代2 = 种子扩展冲 lpips<=0.360')
    else:
        print('[三态] B: 仅 m0s0 系孤点再现不足 — 世代2 = m0s0 续训探底 + w1.8 弱感知世代')


if __name__ == '__main__':
    main()
