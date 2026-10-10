# -*- coding: utf-8 -*-
"""C2 独立复核：不依赖生成脚本的判定函数，从磁盘重算 100 图清单与哈希。

用法: python verify_c2.py <run_dir>
产出: <run_dir>/evidence/c2_verdict.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
TEST_DIR = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集')


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    man = json.loads((run / 'evidence' / 'generation_manifest.json').read_text(encoding='utf-8'))
    outd = run / 'output_dir'
    errors = []
    checked = 0
    key_sets = {}
    sizes = {}
    for rec in man['images']:
        if rec['status'] != 'OK':
            errors.append({'case': rec['case'], 'error': 'status=%s' % rec['status'],
                           'detail': rec.get('error')})
            continue
        i = rec['case']
        p = outd / ('case%d.jpg' % i)
        if not p.is_file():
            errors.append({'case': i, 'error': 'missing_output'})
            continue
        if sha256_file(p) != rec['output_file_sha256']:
            errors.append({'case': i, 'error': 'output_file_sha256_mismatch'})
        dec = np.asarray(Image.open(p).convert('RGB'))
        if hashlib.sha256(dec.tobytes()).hexdigest() != rec['output_decoded_pixels_sha256']:
            errors.append({'case': i, 'error': 'decoded_pixels_mismatch'})
        src = TEST_DIR / ('case%d.jpg' % i)
        if sha256_file(src) != rec['input_file_sha256']:
            errors.append({'case': i, 'error': 'input_file_sha256_mismatch'})
        if [int(dec.shape[1]), int(dec.shape[0])] != rec['input_wh']:
            errors.append({'case': i, 'error': 'size_mismatch',
                           'got': [int(dec.shape[1]), int(dec.shape[0])],
                           'want': rec['input_wh']})
        key_sets.setdefault(tuple(sorted(rec.keys())), []).append(i)
        sizes.setdefault(tuple(rec['input_wh']), 0)
        sizes[tuple(rec['input_wh'])] += 1
        checked += 1
    verdict = {
        'schema': 'EXPORT01-c2-verdict/1',
        'n_images_manifest': len(man['images']),
        'n_ok': man.get('n_ok'),
        'n_checked_from_disk': checked,
        'manifest_status': man['status'],
        'field_schema_uniform': len(key_sets) == 1,
        'n_distinct_field_schemas': len(key_sets),
        'input_size_histogram': {str(k): v for k, v in sizes.items()},
        'errors': errors,
        'bundle': man['bundle'],
        'jpeg_params': man['jpeg_params'],
        'elapsed_s': man.get('elapsed_s'),
        'output_bytes_total': man.get('output_bytes_total'),
        'verdict': 'PASS' if (checked == 100 and not errors and len(key_sets) == 1 and
                              man.get('n_ok') == 100 and man['status'] == 'OK') else 'FAIL',
    }
    (run / 'evidence' / 'c2_verdict.json').write_text(json.dumps(verdict, indent=1,
                                                                 ensure_ascii=False),
                                                      encoding='utf-8')
    print(json.dumps({k: verdict[k] for k in ('verdict', 'n_checked_from_disk', 'n_ok',
                                              'field_schema_uniform', 'n_distinct_field_schemas',
                                              'input_size_histogram', 'errors')},
                     ensure_ascii=False)[:1200])


if __name__ == '__main__':
    main()
