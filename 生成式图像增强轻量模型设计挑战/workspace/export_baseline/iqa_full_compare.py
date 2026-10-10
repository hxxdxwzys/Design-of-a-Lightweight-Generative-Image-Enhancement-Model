# -*- coding: utf-8 -*-
"""全量 NR-IQA 代理对照：A2（42.2145）vs OPT-01（45.0509）各 100 张，逐图三指标 + 配对统计。

用途：判断官方质量项是否对本次改动敏感（若代理完全持平 ⇒ 分数提升主要来自速度项）。
用法: python iqa_full_compare.py <dirA> <dirB> [--n 100]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def main():
    dA, dB = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 100
    import pyiqa
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = {k: pyiqa.create_metric(k, device=dev) for k in ('clipiqa', 'maniqa', 'niqe')}
    files = ['case%d.jpg' % i for i in range(1, n + 1)]
    out = {'schema': 'IQA-full-compare/1', 'dirA': str(dA), 'dirB': str(dB), 'n': n,
           'dirA_label': 'A2_fp16_v1 (official 42.2145)',
           'dirB_label': 'OPT-01 (official 45.0509)', 'rows': []}
    t0 = time.perf_counter()
    for f in files:
        rec = {'file': f}
        for tag, d in (('A', dA), ('B', dB)):
            p = d / f
            if not p.is_file():
                rec[tag] = None
                continue
            vals = {}
            with torch.no_grad():
                for k, m in metrics.items():
                    vals[k] = float(m(str(p)).item())
            rec[tag] = vals
        out['rows'].append(rec)
        if len(out['rows']) % 10 == 0:
            print('  %d/%d  %.0fs' % (len(out['rows']), n, time.perf_counter() - t0), flush=True)
    ok = [r for r in out['rows'] if r.get('A') and r.get('B')]
    summary = {}
    for k in ('clipiqa', 'maniqa', 'niqe'):
        a = np.asarray([r['A'][k] for r in ok])
        b = np.asarray([r['B'][k] for r in ok])
        summary[k] = {'A_mean': float(a.mean()), 'B_mean': float(b.mean()),
                      'A_median': float(np.median(a)), 'B_median': float(np.median(b)),
                      'mean_rel_change_%': float((b.mean() - a.mean()) / a.mean() * 100),
                      'n_B_better': int((b > a).sum()) if k != 'niqe' else int((b < a).sum()),
                      'n': len(ok)}
    out['summary'] = summary
    out['wall_s'] = round(time.perf_counter() - t0, 1)
    p = Path(r'D:\models\iqa_full_compare.json')
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print('written', p, out['wall_s'], 's')


if __name__ == '__main__':
    main()
