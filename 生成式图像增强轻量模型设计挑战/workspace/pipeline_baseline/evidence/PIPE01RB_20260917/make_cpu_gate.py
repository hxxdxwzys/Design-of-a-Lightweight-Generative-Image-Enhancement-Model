# -*- coding: utf-8 -*-
"""PIPE-01R3 CPU 放行门槛生成器（current_task §7.1）。

在**隔离副本**中运行真实测试套件（机器可读 + verbose 两份日志），生成覆盖矩阵与哈希对照，
并据此产出 cpu_gate.json：只有全部门槛通过才写 READY_FOR_RB_ATTEMPT_01。

用法: D:/Python/python.exe make_cpu_gate.py <evidence_dir>
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # evidence/<批次> 目录
PKG = HERE.parents[1]                            # pipeline_baseline
PY = sys.executable
OLD9 = {
    'evidence/cpu_test_log.txt': 'ab69d618af002037',
    'evidence/hashes.json': 'ce805c4bb6425cc1',
    'evidence/key_contract.json': 'c987f5b7a9149af0',
    'evidence/prereg_PIPE01.json': 'a60c0085943b80fc',
    'evidence/summarize_run.py': 'a56395062f3abc8f',
    'evidence/run_PIPE01_20260917_000000/summary.txt': '89582410bfb6a156',
    'evidence/run_PIPE01_20260917_000000/paired_readings.json': 'c2b586cb5efbf795',
    'evidence/run_PIPE01_20260917_000000/gpu_run_log.txt': 'bbced5478ac860d5',
    'evidence/run_PIPE01_20260917_000000/prereg.json': 'a60c0085943b80fc',
}
RA2_7 = {
    'core.py': '1f85ed56652c8070de77a630d55856e9340f6d8c97c1dc2e7fd6ee2f3f3fb597',
    'runner.py': 'f9d82c973580ce69ab498969327b9c9fe09c9c0cac33a7ab754de4d2d87920e2',
    'offline.py': 'e8874ee88fa78f7c2183f3b1a9f8d1f52a26a193b3917e99d717caf0136caada',
    'pair_verify.py': 'b6d5dafb397c8b3e64caf146811d40371649276154d598250a856541c9c329cc',
    'make_evidence.py': 'e00e2ab39d79913d3e70bcc95caca154f869e948a467de434b52b4408976a2c0',
    'config.json': '1a2e44d7709168187b28e30ffa15b35f9f0f5431205ec9b983d50a9c90aca42a',
    'tests/test_pipe01_cpu.py': 'b353e5877d2f3657fe236af1dbdec32f4694a58d592b9f31aae49b677d92124d',
}
SRC7 = ('core.py', 'runner.py', 'offline.py', 'pair_verify.py', 'make_evidence.py',
        'config.json', 'tests/test_pipe01_cpu.py')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(1 << 23)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=3600)
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def main(argv):
    ev = Path(argv[1]) if len(argv) > 1 else HERE
    ev.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    iso = Path(tempfile.mkdtemp(prefix='pipe01_cpu_gate_')) / 'pipeline_baseline'
    shutil.copytree(str(PKG), str(iso),
                    ignore=shutil.ignore_patterns('evidence', '__pycache__', '*.pyc'))
    gate = {'schema': 'PIPE01-cpu-gate/1', 'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'isolated_dir': str(iso), 'pkg': str(PKG), 'open_issues': [], 'gates': []}

    rc1, out1 = run([PY, '-B', str(HERE / 'run_tests_machine.py'), str(iso / 'tests')], iso)
    (ev / 'cpu_test_results.txt').write_text(out1, encoding='utf-8')
    rc2, out2 = run([PY, '-B', str(iso / 'tests' / 'test_pipe01_cpu.py'), '-v'], iso)
    (ev / 'cpu_test_log_isolated.txt').write_text(out2, encoding='utf-8')
    gate['isolated_exit_codes'] = {'machine_runner': rc1, 'unittest_verbose': rc2}

    rc3, out3 = run([PY, '-B', str(HERE / 'make_coverage_matrix.py'),
                     str(ev / 'cpu_test_results.txt'), str(ev / 'coverage_matrix.json')], ev)
    cov = json.loads((ev / 'coverage_matrix.json').read_text(encoding='utf-8')) \
        if (ev / 'coverage_matrix.json').is_file() else {}
    gate['coverage'] = {'ran_tests': cov.get('ran_tests'), 'failures': cov.get('failures'),
                        'errors': cov.get('errors'), 'verdict': cov.get('log_verdict'),
                        'missing': cov.get('missing_tests'), 'unmapped': cov.get('unmapped_tests')}
    for item, v in (cov.get('mapped') or {}).items():
        gate['gates'].append({'gate': item, 'tests': [t['test'] for t in v['tests']],
                              'all_present_ok': v['all_present_ok']})

    src = {}
    for rel in SRC7:
        p = PKG / rel
        src[rel] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
    gate['source_hashes_frozen'] = src
    gate['ra2_source_hashes'] = RA2_7
    gate['ra2_unchanged'] = {k: (src[k]['sha256'] == v) for k, v in RA2_7.items()}
    old = {}
    for rel, want in OLD9.items():
        p = PKG / rel
        old[rel] = {'expected_prefix': want,
                    'actual_prefix': sha256(p)[:16] if p.is_file() else None}
    gate['old_evidence_check'] = old
    gate['old_evidence_unchanged'] = all(v['expected_prefix'] == v['actual_prefix']
                                         for v in old.values())

    checks = {
        'machine_runner_exit_0': rc1 == 0,
        'unittest_verbose_exit_0': rc2 == 0,
        'no_failures': cov.get('failures') == 0,
        'no_errors': cov.get('errors') == 0,
        'coverage_missing_empty': not cov.get('missing_tests'),
        'coverage_all_gates_ok': all(g['all_present_ok'] for g in gate['gates']),
        'old_evidence_unchanged': gate['old_evidence_unchanged'],
        'config_unchanged_vs_ra2': src['config.json']['sha256'] == RA2_7['config.json'],
        'runner_unchanged_vs_ra2': src['runner.py']['sha256'] == RA2_7['runner.py'],
    }
    gate['checks'] = checks
    gate['verdict'] = ('READY_FOR_RB_ATTEMPT_01' if all(checks.values()) else 'NOT_READY')
    gate['elapsed_s'] = round(time.time() - t0, 1)
    if not all(checks.values()):
        gate['open_issues'].append('存在未通过门槛: %s'
                                   % [k for k, v in checks.items() if not v])
    (ev / 'cpu_gate.json').write_text(json.dumps(gate, ensure_ascii=False, indent=1),
                                      encoding='utf-8')
    print('[cpu-gate] verdict=%s checks=%s' % (gate['verdict'],
                                               {k: bool(v) for k, v in checks.items()}))
    print('[cpu-gate] tests=%s failures=%s errors=%s missing=%s' %
          (cov.get('ran_tests'), cov.get('failures'), cov.get('errors'),
           len(cov.get('missing_tests') or [])))
    shutil.rmtree(str(iso.parent), ignore_errors=True)
    return 0 if gate['verdict'] == 'READY_FOR_RB_ATTEMPT_01' else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
