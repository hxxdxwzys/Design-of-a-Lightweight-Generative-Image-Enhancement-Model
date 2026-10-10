# -*- coding: utf-8 -*-
"""按 selection_analysis.json 的策略逐图装配输出目录 (文件复制, 无渲染)
用法: D:/Python/python.exe assemble_mix.py --strategy choice_safe_M0.02 --out D:/models/cnn_out_mix
"""
import argparse
import json
import shutil
from pathlib import Path

SWEEP = Path(r'D:/models/sweep_test_out')
SRC_MAP = {
    'old_s6000': Path(r'D:/models/cnn_out_s6k'),
    'f2k_s4000': Path(r'D:/models/cnn_out_f2k'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='choice_safe_M0.02',
                    help='choice_safe_M0.02 / choice_all_M0.02 / ceiling_safe.per_image / ceiling_all.per_image / hybrid')
    ap.add_argument('--out', default=r'D:/models/cnn_out_mix')
    args = ap.parse_args()

    if args.strategy == 'hybrid':
        choice = json.load(open(r'D:/tmp_timing/hybrid_choice.json'))['choice']
    elif args.strategy.startswith('map:'):
        margin = args.strategy.split(':', 1)[1]
        choice = json.load(open(r'D:/tmp_timing/choice_maps.json'))['maps']['M' + margin]['choice']
    else:
        sel = json.load(open(r'D:/tmp_timing/selection_analysis.json'))
        if args.strategy in sel:
            choice = sel[args.strategy]
        else:
            k, sub = args.strategy.split('.', 1)
            choice = sel[k][sub]
    print(f'[strategy] {args.strategy}: {len(choice)} 张图', flush=True)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    from collections import Counter
    cnt = Counter()
    for k, ckpt in choice.items():
        src = SRC_MAP.get(ckpt, SWEEP / ckpt)
        p = src / f'{k}.jpg'
        assert p.exists(), f'missing {p}'
        shutil.copy2(p, out / f'{k}.jpg')
        cnt[ckpt] += 1
    print('[sources]', dict(cnt.most_common()), flush=True)
    print(f'[ok] {out} ({len(choice)} jpgs)', flush=True)


if __name__ == '__main__':
    main()
