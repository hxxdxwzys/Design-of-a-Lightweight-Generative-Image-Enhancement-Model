# -*- coding: utf-8 -*-
"""PIPE-01 读数汇总（只读）：把某个 run 目录的 paired_readings.json 摘要成可审阅文本。

用法: D:/Python/python.exe summarize_run.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv):
    run = Path(argv[1] if len(argv) > 1 else
               r'D:/models/pipeline_baseline/PIPE01_20260917_000000')
    d = json.loads((run / 'paired_readings.json').read_text(encoding='utf-8'))
    inst = d['instances']
    out = []

    class _T:
        def write(self, s):
            out.append(s)

    real = sys.stdout
    sys.stdout = _T()
    try:
        rc = _dump(run, d, inst)
    finally:
        txt = ''.join(out)
        sys.stdout = real
        (run / 'summary.txt').write_text(txt, encoding='utf-8')
        print('WROTE %s (%d bytes, utf-8)' % (run / 'summary.txt', len(txt.encode('utf-8'))))
    return rc


def _dump(run, d, inst):
    print('run_dir        %s' % run)
    print('status         %s   criteria_all_pass=%s   device=%s'
          % (d['status'], d.get('criteria_all_pass'), inst.get('device')))
    print('forwards       %s   elapsed_s=%.1f' % (d['forwards'], d['elapsed_s']))
    print('prereg_sha256  %s' % inst.get('prereg_sha256'))
    print('load           A=%.1fs peak=%s MiB | B=%.1fs peak=%s MiB'
          % (inst['A'][0]['wall_s'], (inst['A_peak_vram_bytes'] or 0) // 2 ** 20,
             inst['B'][0]['wall_s'], (inst['B_peak_vram_bytes'] or 0) // 2 ** 20))
    print('--- 逐样本成对读数 ---')
    for k, v in d['pairs'].items():
        print(' %-14s max_abs=%.3e mean_abs=%.3e n=%d png_identical=%s png=%s'
              % (k, v['max_abs_diff'], v['mean_abs_diff'], v['n'], v['png_identical'],
                 v['png_shape']))
    print('--- 同实例重复 512 ---')
    for k, v in (d.get('repeat_512') or {}).items():
        print(' %-18s max_abs=%.3e mean_abs=%.3e' % (k, v['max_abs_diff'], v['mean_abs_diff']))
    print('--- 逐次前向 ---')
    for r in d['records']:
        shape = r.get('shape') or r.get('tensor_shape')
        mn = r.get('min', r.get('tensor_min'))
        mx = r.get('max', r.get('tensor_max'))
        fin = r.get('finite')
        print(' %s %-14s wall=%5.1fs shape=%s min=%s max=%s finite=%s png_bytes=%s'
              % (r['instance'], r['tag'], r['wall_s'], shape, mn, mx, fin,
                 r.get('png_bytes')))
    i = inst.get('A_identity', {})
    e = i.get('effective_fingerprint', {})
    print('--- 身份链 ---')
    print(' config_sha256  %s' % i.get('config_sha256'))
    if e:
        print(' eff_model      %s  n_params=%s'
              % (e['model']['digest_sha256'], e['model']['n_params']))
        print(' eff_vae        %s' % e['vae']['digest_sha256'])
        print(' eff_decoder    %s' % e['light_decoder']['digest_sha256'])
    print(' load_report    %s' % json.dumps(i.get('load_report'), ensure_ascii=False))
    print(' tile_report    %s' % json.dumps(i.get('tile_report'), ensure_ascii=False))
    print('--- 输入 ---')
    for k, v in d['inputs'].items():
        print(' %-14s shape=%s sha=%s crop_box=%s'
              % (k, v['tensor_shape'], v['tensor_sha256'][:16], v['crop_box']))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
