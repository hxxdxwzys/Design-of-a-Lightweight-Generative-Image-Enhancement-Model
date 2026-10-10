# -*- coding: utf-8 -*-
"""M26 再选择混合组装 (2026-08-26): 逐图 choice map -> 从池目录复制 jpg -> 输出目录
用法:
  D:/Python/python.exe b_2x_mix_assemble.py --rule steep_v6 --out D:/models/cnn_out_mix_m26
  --rule steep_v6:  v7 clip<0.905 的图 -> vosr_lora_v6_s4000, 其余 v7_seed8
  --rule steep_v6_topN:N: 仅 capped gain 前 N 大的图换 v6
  --rule u2_threshold:T:   1x clip >= T 的图用 2x 输出 (test_out_2x), 其余 v7_seed8
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

POOL = Path(r'D:/models/sweep_test_out')
V7 = 'vosr_lora_v7_s5000_sc0.8_seed8'
V6 = 'vosr_lora_v6_s4000'
SR = json.load(open(POOL / 'sweep_results.json', encoding='utf-8'))['results']
by_name = {r['name']: r for r in SR}
pi_v7 = by_name[V7]['per_image']
pi_v6 = by_name[V6]['per_image']


def case_keys():
    return sorted(pi_v7, key=lambda k: int(k.replace('case', '')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rule', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f'[warn] {out} exists, removing', flush=True)
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rule = args.rule
    if rule == 'steep_v6':
        sel = {k: (V6 if float(pi_v7[k]['clipiqa']) < 0.905 else V7) for k in case_keys()}
    elif rule.startswith('steep_v6_topN'):
        n = int(rule.split(':')[1])
        gains = sorted(case_keys(), key=lambda k: float(pi_v6[k]['clipiqa']) - float(pi_v7[k]['clipiqa']), reverse=True)
        sel = {k: V7 for k in case_keys()}
        for k in gains[:n]:
            sel[k] = V6
    elif rule.startswith('u2_threshold'):
        t = float(rule.split(':')[1])
        u2 = Path(r'D:/models/test_out_2x')
        sel = {}
        for k in case_keys():
            if float(pi_v7[k]['clipiqa']) >= t and (u2 / (k + '.png')).exists():
                sel[k] = 'u2'
            else:
                sel[k] = V7
    else:
        raise SystemExit(f'unknown rule {rule}')

    n_v6 = sum(1 for v in sel.values() if v == V6)
    n_u2 = sum(1 for v in sel.values() if v == 'u2')
    for k, src in sel.items():
        if src == 'u2':
            p = Path(r'D:/models/test_out_2x') / (k + '.png')
            with Image.open(p) as im:
                w, h = im.size
                im2 = im.resize((w // 2, h // 2), Image.BICUBIC)
                im2.save(out / (k + '.jpg'), quality=98)
        else:
            shutil.copy2(POOL / src / (k + '.jpg'), out / (k + '.jpg'))
    print(f'[ok] {out}: 100 jpgs (v6={n_v6} u2={n_u2} v7={100 - n_v6 - n_u2})', flush=True)


if __name__ == '__main__':
    main()
