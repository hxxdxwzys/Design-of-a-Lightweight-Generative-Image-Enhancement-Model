# -*- coding: utf-8 -*-
"""夜间链 v2 (2026-09-14 夜): v7 收尾 -> v7 productize -> HYPIR 蒸馏数据 -> HYPIR-GAN 训练 -> productize -> (余量)v8 宽模型
理由: 排行榜证明新机制下 86~124 可达 => 差距在图像质量项 => 唯一没换过的变量是【图像源】
日志: D:/tmp_timing/night_chain2.log
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, r'D:/tmp_timing')
from gan_chain import run, productize, log, wait_marker  # 复用 (其 log 写 gan_chain.log)

PY = r'D:/Python/python.exe'
T = Path(r'D:/tmp_timing')
LOG2 = T / 'night_chain2.log'


def log2(m):
    s = f'[{time.strftime("%m-%d %H:%M:%S")}] {m}'
    print(s, flush=True)
    with LOG2.open('a', encoding='utf-8') as f:
        f.write(s + '\n')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    log2('=' * 70)
    log2('NIGHT CHAIN2: v7 -> HYPIR 蒸馏 -> (余量) v8 宽模型')
    res = []
    wait_marker(Path(r'D:/models/tiny_v1/train_gan.log'), 'TRAIN_GAN_DONE', 3 * 3600)
    ck = Path(r'D:/models/tiny_v1/v7gan_s25000.pt')
    if ck.exists():
        res.append(productize(ck, 'TinyV7gan', 2, '0.87M GAN蒸馏(M24目标) 256域'))

    run('HYPIR 蒸馏数据', [PY, '-u', str(T / 'gen_distill_hypir.py')])
    n = len(list((Path(r'D:/datasets/distill_hypir/lq')).glob('*.jpg')))
    log2(f'HYPIR 对: {n}')
    if n >= 800:
        run('train v10 (HYPIR-GAN)', [PY, '-u', str(T / 'train_tiny_gan.py'),
                                      '--distill_dir', r'D:/datasets/distill_hypir',
                                      '--steps', '20000', '--bs', '8', '--tag', 'v10hypir'])
        ck = Path(r'D:/models/tiny_v1/v10hypir_s20000.pt')
        if ck.exists():
            res.append(productize(ck, 'TinyV10hypir', 2, '0.87M GAN蒸馏(HYPIR目标) 256域'))
    run('train v8 (3.5M + GAN)', [PY, '-u', str(T / 'train_tiny_gan.py'), '--steps', '20000',
                                  '--bs', '4', '--ch', '64,64,64,128', '--tag', 'v8gan'])
    ck = Path(r'D:/models/tiny_v1/v8gan_s20000.pt')
    if ck.exists():
        res.append(productize(ck, 'TinyV8gan', 2, '3.5M GAN蒸馏(M24目标) 256域'))
    with (T / 'NIGHT2_REPORT.md').open('w', encoding='utf-8') as f:
        f.write('# 夜间链 v2 报告 (2026-09-14)\n\n')
        f.write('参照: M24 clipiqa 0.9023/lapvar 508 -> 43.6951 | v5 0.3897/92 -> 33.0382 | HYPIR 图 clipiqa 0.4478\n\n')
        f.write('| 候选 | 目标源 | 质量代理 | 时延 | 归位 |\n|---|---|---|---|---|\n')
        for r in res:
            f.write(f"| {r['tag']} | {r['note']} | {r['metric']} | {r['latency']} | {r['packed']} |\n")
    log2('NIGHT_CHAIN2_DONE')


if __name__ == '__main__':
    main()
