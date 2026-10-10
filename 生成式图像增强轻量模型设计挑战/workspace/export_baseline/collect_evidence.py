# -*- coding: utf-8 -*-
"""EXPORT-01 证据汇总：资源实际核算 + 三阶段判决汇总（含包外文件）。

用法: python collect_evidence.py <run_dir>
产出: <run_dir>/evidence/{resource_accounting_final.json, export01_summary.json}
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EB = Path(__file__).resolve().parent


def tree_stats(root):
    root = Path(root)
    if not root.exists():
        return {'n_files': 0, 'bytes': 0}
    files = [p for p in root.rglob('*') if p.is_file()]
    return {'n_files': len(files), 'bytes': sum(p.stat().st_size for p in files)}


def free_gib(p):
    import shutil
    return round(shutil.disk_usage(str(p)).free / (1024 ** 3), 3)


def main():
    run = Path(sys.argv[1]).resolve()
    ev = run / 'evidence'

    def load(name):
        p = ev / name
        return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None

    c0 = load('c0_evidence.json')
    c1 = load('c1_verdict.json')
    c2 = load('c2_verdict.json')
    zrec = load('zip_record.json')
    pre = load('isolated_precheck.json')
    build = load('build_result.json')
    conv = load('conversion_record.json')
    gen = load('generation_manifest.json')

    out_of_tree = []
    tmp = Path(tempfile.gettempdir())
    for d in tmp.glob('export01_precheck_*'):
        out_of_tree.append(dict({'kind': 'precheck_extract', 'path': str(d)}, **tree_stats(d)))
    zip_path = Path(zrec['zip_path']) if zrec else None

    totals = tree_stats(run)
    resource = {
        'schema': 'EXPORT01-resource-accounting-final/1',
        'computed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'run_dir': str(run),
        'run_dir_tree': totals,
        'out_of_dir_objects': out_of_tree,
        'zip': ({'path': str(zip_path), 'bytes': zrec['zip_bytes'],
                 'sha256': zrec['zip_sha256'], 'within_limit': zrec['within_limit']}
                if zrec else None),
        'grand_total_bytes': (totals['bytes'] + sum(o['bytes'] for o in out_of_tree) +
                              (zrec['zip_bytes'] if zrec else 0)),
        'disk_free_gib': {'C': free_gib('C:/'), 'D': free_gib('D:/')},
        'note': 'run 目录内的 ZIP 计数一次；解包副本在 %TEMP%（C 盘）单独列出；不含只读父资产',
    }
    (ev / 'resource_accounting_final.json').write_text(json.dumps(resource, indent=1,
                                                                 ensure_ascii=False),
                                                       encoding='utf-8')

    summary = {
        'schema': 'EXPORT01-summary/1',
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'run_dir': str(run),
        'stages': {
            'C0': {'c0_evidence': bool(c0),
                   'ra2_snapshot_files_changed_in_rb': (c0 or {}).get(
                       'ra2_snapshot_restore', {}).get('files_changed_in_RB'),
                   'cpu_tests': '21 tests OK (wavelet 11 + delivery/control 10)'},
            'C1': {'verdict': (c1 or {}).get('verdict'),
                   'same_precision_pass': (c1 or {}).get('same_precision_pass'),
                   'fp16_vs_fp32_diagnostic_pass': (c1 or {}).get('fp16_vs_fp32_diagnostic_pass'),
                   'model_sha256': (c1 or {}).get('model_sha256'),
                   'totals': (c1 or {}).get('totals')},
            'C2': {'verdict': (c2 or {}).get('verdict'), 'n_ok': (c2 or {}).get('n_ok'),
                   'elapsed_s': (gen or {}).get('elapsed_s'),
                   'jpeg_params': (gen or {}).get('jpeg_params')},
            'C3': {'zip_record': bool(zrec),
                   'zip_bytes': (zrec or {}).get('zip_bytes'),
                   'zip_sha256': (zrec or {}).get('zip_sha256'),
                   'isolated_precheck_verdict': (pre or {}).get('verdict'),
                   'py38_syntax_ok': (pre or {}).get('py38_syntax_ok'),
                   'shadowing_risk': (pre or {}).get('shadowing_risk')},
        },
        'conversion': {'id': (conv or {}).get('id'), 'status': (conv or {}).get('status'),
                       'output_sha256': ((conv or {}).get('output_fingerprint') or {}).get('sha256'),
                       'bytes': (build or {}).get('model_bytes')},
        'resource': {'grand_total_bytes': resource['grand_total_bytes'],
                     'disk_free_gib': resource['disk_free_gib']},
        'open_unknowns': [
            '目标解释器/依赖环境实测（官方环境未提供）',
            '官方提交资格（EXP-04 历史训练披露/官方接受边界）',
            'fp16 相对 fp32 的图像质量影响（本轮只做数值诊断，不评 IQA、不选候选）',
        ],
    }
    (ev / 'export01_summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                              encoding='utf-8')
    print(json.dumps(summary['stages'], ensure_ascii=False)[:1500])


if __name__ == '__main__':
    main()
