# -*- coding: utf-8 -*-
"""生成 phase-1 源参考产物的来源登记（Python 直接写 UTF-8，无 BOM）。

用法: python make_phase1_provenance.py <run_dir>
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    src = run / 'evidence' / 'src_outputs'
    files = {}
    for p in sorted(src.glob('*_src.npy')):
        st = p.stat()
        files[p.name] = {'sha256': sha256_file(p), 'bytes': st.st_size,
                         'written_at': time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(st.st_mtime))}
    prov = {
        'note': '源参考（fp16 eager + 同一 delivery_runner 编排）的四个输出取自本预登记下的一次尝试的 '
                'phase-1；该次在 phase-2 首例因设备不匹配崩溃，未产生任何判定，phase-1 产物完整且按哈希固定。'
                '本次只执行重载侧，避免重复消耗 4K 前向额度（复用允许，来源如实登记）。',
        'aborted_attempt_log': 'evidence_verify_c1_stdout2.txt',
        'phase1_timings_s': {'case1_crop512': 0.75, 'case1_full': 24.54,
                             'case7_full': 24.19, 'case1_crop511x513': 0.62},
        'runner_recipe_sha256': sha256_file(run / 'model_dir' / 'models_export.json'),
        'model_sha256': sha256_file(run / 'model_dir' / 'your_model.pt'),
        'files': files,
    }
    out = src / 'phase1_provenance.json'
    out.write_text(json.dumps(prov, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'out': str(out), 'n_files': len(files),
                      'bytes': out.stat().st_size, 'model_sha256': prov['model_sha256'][:16]},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
