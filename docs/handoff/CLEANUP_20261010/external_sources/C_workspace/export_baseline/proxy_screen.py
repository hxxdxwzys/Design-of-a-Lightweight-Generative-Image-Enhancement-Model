# -*- coding: utf-8 -*-
"""PROXY-SCREEN：用官方风格 NR 指标对「历史候选目录」做一致性重评（同文件名子集）。

用法: python proxy_screen.py <out_json> <label=dir> [<label=dir> ...] [--musiq-n 3]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
    out_json = Path(sys.argv[1])
    pairs = []
    for a in sys.argv[2:]:
        if a.startswith('--') or '=' not in a:
            continue
        label, d = a.split('=', 1)
        p = Path(d)
        if p.is_dir():
            pairs.append((label, p))
    musiq_n = int(sys.argv[sys.argv.index('--musiq-n') + 1]) if '--musiq-n' in sys.argv else 3
    # 公共文件名（按 stem 归一：case1.png / case1.jpg 视为同一 case）
    sets = {}
    for label, d in pairs:
        sets[label] = {q.stem: q for q in sorted(d.iterdir()) if q.suffix.lower() in EXTS}
    common = set.intersection(*[set(v) for v in sets.values()]) if sets else set()
    common = sorted(common, key=lambda s: int(''.join(ch for ch in s if ch.isdigit()) or 0))
    print('labels', [l for l, _ in pairs], 'common', len(common), common[:8], flush=True)
    import torch
    import pyiqa
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = {k: pyiqa.create_metric(k, device=dev) for k in ('clipiqa', 'maniqa', 'niqe')}
    if musiq_n > 0:
        metrics['musiq'] = pyiqa.create_metric('musiq', device=dev)
    out = {'schema': 'PROXY-screen/1', 'metrics': list(metrics), 'n_common': len(common),
           'cases': common, 'rows': {}}
    t0 = time.perf_counter()
    for label, d in pairs:
        vals = {k: [] for k in metrics}
        per_case = {}
        for i, stem in enumerate(common):
            p = sets[label][stem]
            got = {}
            with torch.no_grad():
                for k, m in metrics.items():
                    if k == 'musiq' and i >= musiq_n:
                        continue
                    v = float(m(str(p)).item())
                    vals[k].append(v)
                    got[k] = round(v, 4)
            per_case[stem] = got
        out['rows'][label] = {'dir': str(d), 'per_case': per_case,
                              'mean': {k: (float(np.mean(v)) if v else None)
                                       for k, v in vals.items()},
                              'n': {k: len(v) for k, v in vals.items()}}
        print('%-26s %s' % (label, {k: (round(v, 4) if v is not None else None)
                                    for k, v in out['rows'][label]['mean'].items()}), flush=True)
    # ── 预登记复合代理：各指标在本次候选间做 min-max 归一（niqe 取反），复合分 = 均值 ──
    labels = [l for l, _ in pairs]
    sign = {'clipiqa': 1.0, 'musiq': 1.0, 'maniqa': 1.0, 'niqe': -1.0}
    norm = {}
    for k in metrics:
        vals = {l: out['rows'][l]['mean'][k] for l in labels if out['rows'][l]['mean'][k] is not None}
        if len(vals) < 2:
            continue
        lo, hi = min(vals.values()), max(vals.values())
        rng = (hi - lo) or 1.0
        norm[k] = {l: sign[k] * (v - lo) / rng for l, v in vals.items()}
    comp = {}
    for l in labels:
        ks = [k for k in norm if l in norm[k]]
        if ks:
            comp[l] = float(np.mean([norm[k][l] for k in ks]))
    out['composite'] = comp
    out['composite_metrics'] = sorted(norm)
    out['ranking'] = sorted(comp, key=lambda l: -comp[l])
    out['wall_s'] = round(time.perf_counter() - t0, 1)
    print('RANKING', json.dumps({l: round(comp[l], 4) for l in out['ranking']}), flush=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print('written', out_json, out['wall_s'], 's')


if __name__ == '__main__':
    main()
