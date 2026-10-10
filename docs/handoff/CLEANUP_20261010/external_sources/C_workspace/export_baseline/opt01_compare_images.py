# -*- coding: utf-8 -*-
"""OPT-01 逐图比对：与 A2 的 100 张 JPEG 用同一解码器解码后比较 RGB 差异幅度。

用法: python opt01_compare_images.py <opt01_run_dir> <a2_run_dir>
说明：差异幅度是**输出变化量**，本地无法据此判断质量好坏；用于向用户如实披露取舍。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    new_run, old_run = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    new_out, old_out = new_run / 'output_dir', old_run / 'output_dir'
    per = []
    for i in range(1, 101):
        a = new_out / ('case%d.jpg' % i)
        b = old_out / ('case%d.jpg' % i)
        rec = {'case': i, 'new_exists': a.is_file(), 'old_exists': b.is_file()}
        if not (a.is_file() and b.is_file()):
            per.append(rec)
            continue
        rec['new_bytes'] = a.stat().st_size
        rec['old_bytes'] = b.stat().st_size
        rec['byte_identical'] = bool(sha256_file(a) == sha256_file(b))
        na = np.asarray(Image.open(a).convert('RGB')).astype(np.int16)
        nb = np.asarray(Image.open(b).convert('RGB')).astype(np.int16)
        rec['same_shape'] = bool(na.shape == nb.shape)
        diff = np.abs(na - nb)
        rec['mean_abs'] = float(diff.mean())
        rec['max_abs'] = int(diff.max())
        rec['frac_pixels_any_diff'] = float((diff.max(axis=2) > 0).mean())
        for thr in (2, 4, 8):
            rec['n_channel_ge%d' % thr] = int((diff >= thr).sum())
        per.append(rec)
    ok = [r for r in per if r.get('mean_abs') is not None]
    means = sorted(r['mean_abs'] for r in ok)
    fracs = sorted(r['frac_pixels_any_diff'] for r in ok)
    out = {'schema': 'OPT01-image-compare/1', 'n_images': len(per), 'n_compared': len(ok),
           'n_byte_identical': sum(1 for r in ok if r['byte_identical']),
           'mean_abs_median': float(np.median(means)), 'mean_abs_p90': float(np.percentile(means, 90)),
           'mean_abs_max': float(max(means)), 'frac_any_diff_median': float(np.median(fracs)),
           'frac_any_diff_p90': float(np.percentile(fracs, 90)),
           'max_abs_over_images': int(max(r['max_abs'] for r in ok)),
           'per_image': per}
    p = new_run / 'evidence' / 'opt01_image_compare.json'
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('n_compared', 'n_byte_identical', 'mean_abs_median',
                                          'mean_abs_p90', 'mean_abs_max', 'frac_any_diff_median',
                                          'frac_any_diff_p90', 'max_abs_over_images')},
                     ensure_ascii=False))
    print('written', p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
