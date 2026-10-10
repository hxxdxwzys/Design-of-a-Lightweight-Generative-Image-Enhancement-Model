# -*- coding: utf-8 -*-
"""IQA 代理能力冒烟：用 pyiqa 计算官方风格 NR 指标（CLIPIQA / MUSIQ / MANIQA / NIQE / BRISQUE）。

用法: python iqa_probe.py <img1> <img2> ... [--limit N]
说明：仅本地测量；不修改任何交付件。
"""
import sys
import time
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    limit = None
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit') + 1])
    imgs = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            imgs.extend(sorted([q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS]))
        else:
            imgs.append(p)
    if limit:
        imgs = imgs[:limit]
    print('images', len(imgs), flush=True)
    import torch
    import pyiqa
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('pyiqa', pyiqa.__version__ if hasattr(pyiqa, '__version__') else '?', 'device', dev, flush=True)
    names = ['clipiqa', 'musiq', 'maniqa', 'niqe', 'brisque']
    metrics = {}
    for n in names:
        t0 = time.perf_counter()
        try:
            metrics[n] = pyiqa.create_metric(n, device=dev)
            print('  metric %-8s ready %.1fs' % (n, time.perf_counter() - t0), flush=True)
        except Exception as e:
            print('  metric %-8s FAILED %s: %s' % (n, type(e).__name__, str(e)[:160]), flush=True)
    rows = []
    for p in imgs:
        row = {'file': str(p)}
        for n, m in metrics.items():
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    v = float(m(str(p)).item())
                row[n] = v
                row[n + '_s'] = round(time.perf_counter() - t0, 2)
            except Exception as e:
                row[n] = None
                row[n + '_err'] = '%s: %s' % (type(e).__name__, str(e)[:80])
        rows.append(row)
        print('%-50s %s' % (Path(p).name, {k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in row.items() if k != 'file'}), flush=True)
    import json
    out = Path(r'D:\models\iqa_probe.json')
    out.write_text(json.dumps({'device': str(dev), 'rows': rows}, indent=1, ensure_ascii=False),
                   encoding='utf-8')
    print('written', out)


if __name__ == '__main__':
    main()
