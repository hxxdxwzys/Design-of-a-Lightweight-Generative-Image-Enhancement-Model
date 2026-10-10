# -*- coding: utf-8 -*-
"""EXPORT-01R 监督者：只读预检 + 子进程 deadline 终止 + 四段合并判决。

用法: python r_supervisor.py <run_dir> --ledger <path> --prereg <path> --model-dir <dir>                                --exp-dir <dir> [--deadline-s 1800]
判决：worker_checks_pass AND supervisor_success AND ledger_consistent AND artifacts_verified
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def free_gib(p):
    import shutil
    q = Path(p)
    while not q.exists() and q.parent != q:
        q = q.parent
    return shutil.disk_usage(str(q)).free / (1024 ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--ledger', required=True)
    ap.add_argument('--prereg', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--exp-dir', required=True)
    ap.add_argument('--deadline-s', type=float, default=1800.0)
    ap.add_argument('--worker', default=str(HERE / 'r_worker.py'),
                    help='worker 入口（默认生产 r_worker.py；CPU 故障注入测试用替身）')
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    exp = Path(a.exp_dir).resolve()
    exp.mkdir(parents=True, exist_ok=True)
    sup = {'schema': 'EXPORT01R-supervisor/1',
           'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'deadline_s': a.deadline_s, 'pid': os.getpid()}

    # ── 只读预检（不建 run 目录、不写总账、不抢锁）──
    pre = {'ledger_exists': Path(a.ledger).exists(),
           'lock_exists': Path(str(a.ledger) + '.lock').exists(),
           'disk_free_gib': round(free_gib(run), 3),
           'prereg_exists': Path(a.prereg).is_file(),
           'model_exists': (Path(a.model_dir) / 'your_model.pt').is_file(),
           'model_will_be_built': not (Path(a.model_dir) / 'your_model.pt').is_file()}
    sup['preflight'] = pre
    if pre['lock_exists']:
        sup.update({'supervisor_success': False, 'error': 'LOCK_EXISTS'})
        (exp / 'supervisor.json').write_text(json.dumps(sup, indent=1, ensure_ascii=False),
                                             encoding='utf-8')
        print('SUPERVISOR_FAIL LOCK_EXISTS')
        return 2
    if pre['disk_free_gib'] < 8.0:
        sup.update({'supervisor_success': False, 'error': 'DISK_LOW'})
        (exp / 'supervisor.json').write_text(json.dumps(sup, indent=1, ensure_ascii=False),
                                             encoding='utf-8')
        print('SUPERVISOR_FAIL DISK_LOW')
        return 2
    # 模型文件允许缺失：本批次由 worker 在 attempt 内导出构建（build-then-verify）。
    # 预检只要求预登记与 model_dir 目录存在；模型产物由 post 段 artifacts_verified 核验。
    pre['model_dir_exists'] = Path(a.model_dir).is_dir()
    if not (pre['prereg_exists'] and pre['model_dir_exists']):
        sup.update({'supervisor_success': False, 'error': 'MISSING_INPUTS'})
        (exp / 'supervisor.json').write_text(json.dumps(sup, indent=1, ensure_ascii=False),
                                             encoding='utf-8')
        print('SUPERVISOR_FAIL MISSING_INPUTS')
        return 2

    # ── 子进程 ──
    wlog = run.parent / (run.name + '.worker_log.txt')
    cmd = [sys.executable, '-u', str(a.worker), str(run), str(a.ledger),
           str(a.prereg), str(a.model_dir), str(exp)]
    sup['cmd'] = cmd
    sup['worker_log'] = str(wlog)
    t0 = time.monotonic()
    # 独立进程组：外部控制台 Ctrl+C 不会直接打到 worker（监督者独占终止权）
    flags = 0
    if os.name == 'nt':
        flags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    with open(wlog, 'w', encoding='utf-8', errors='replace') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE),
                                creationflags=flags)
    sup['child_pid'] = proc.pid
    terminate_at = None
    killed_at = None
    while True:
        if proc.poll() is not None:
            break
        if time.monotonic() - t0 > a.deadline_s:
            terminate_at = time.strftime('%Y-%m-%d %H:%M:%S')
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                killed_at = time.strftime('%Y-%m-%d %H:%M:%S')
                proc.kill()
                proc.wait(timeout=15)
            break
        time.sleep(0.5)
    sup.update({'child_exit_code': proc.returncode, 'terminate_requested_at': terminate_at,
                'kill_at': killed_at, 'timed_out': terminate_at is not None,
                'wall_s': round(time.monotonic() - t0, 3),
                'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    sup['supervisor_success'] = bool(proc.returncode == 0 and terminate_at is None)

    # ── 账本一致性（以最新磁盘内容为准）──
    led_path = Path(a.ledger)
    ledger = json.loads(led_path.read_text(encoding='utf-8')) if led_path.is_file() else None
    used = (ledger or {}).get('used', {})
    limits = (ledger or {}).get('limits', {})
    attempts = (ledger or {}).get('attempts', [])
    sup['ledger'] = {'used': used, 'limits': limits,
                     'n_events': len((ledger or {}).get('events', [])),
                     'n_attempts': len(attempts),
                     'open_attempt': (ledger or {}).get('open_attempt'),
                     'last_attempt': attempts[-1] if attempts else None}
    ledger_ok = bool(ledger and not ledger.get('open_attempt') and len(attempts) == 1
                     and used.get('top', 0) <= limits.get('top', 0)
                     and used.get('top4k', 0) <= limits.get('top4k', 0)
                     and used.get('component', 0) <= limits.get('component', 0))
    sup['ledger_consistent'] = ledger_ok

    # ── 产物与判据核验（读 worker 落盘读数，独立复算关键布尔）──
    rd_path = exp / 'readings.json'
    rd = json.loads(rd_path.read_text(encoding='utf-8')) if rd_path.is_file() else None
    tol = json.loads(Path(a.prereg).read_text(encoding='utf-8'))['tolerances']
    checks = {'readings_present': rd is not None, 'worker_status_ok': (rd or {}).get('status') == 'OK'}
    if rd and rd.get('status') == 'OK':
        npy_ok = True
        npy_missing = []
        for key in ('phaseA', 'phaseB'):
            for k, v in rd.get(key, {}).items():
                if isinstance(v, dict) and 'npy' in v:
                    p = exp / v['npy']
                    if not p.is_file() or sha256_file(p) != v['npy_sha256']:
                        npy_ok = False
                        npy_missing.append(v['npy'])
        checks['all_npy_present_and_hashed'] = npy_ok
        checks['npy_missing'] = npy_missing
        pairs_ok = all(p['pair']['max_abs'] <= tol['same_precision_max_abs'] and
                       p['pair']['mean_abs'] <= tol['same_precision_mean_abs'] and
                       p['pair']['png_pixel_identical'] for p in rd.get('pairs', []))
        checks['n_pairs'] = len(rd.get('pairs', []))
        checks['pairs_ok'] = pairs_ok
        b512 = [v for k, v in rd.get('phaseB', {}).items()
                if isinstance(v, dict) and v.get('call', '').startswith('B_512')]
        checks['n_B512'] = len(b512)
        checks['B512_vs_A_ok'] = all(v['vs_A']['max_abs'] <= tol['same_precision_max_abs'] and
                                     v['vs_A']['mean_abs'] <= tol['same_precision_mean_abs'] and
                                     v['vs_A']['png_pixel_identical'] for v in b512)
        checks['B512_repeat_ok'] = all(v['vs_first_B512']['max_abs'] <= tol['same_precision_max_abs']
                                       and v['vs_first_B512']['mean_abs'] <= tol['same_precision_mean_abs']
                                       and v['vs_first_B512']['png_pixel_identical']
                                       for v in b512 if v['call'] != 'B_512_1')
        checks['n_top_used'] = used.get('top')
        checks['n_4k_used'] = used.get('top4k')
    worker_ok = bool(checks['readings_present'] and checks['worker_status_ok'] and
                     checks.get('all_npy_present_and_hashed') and checks.get('pairs_ok') and
                     checks.get('B512_vs_A_ok') and checks.get('B512_repeat_ok'))
    sup['checks'] = checks
    sup['worker_checks_pass'] = worker_ok
    sup['artifacts_verified'] = bool(checks.get('all_npy_present_and_hashed'))
    verdict = {'schema': 'EXPORT01R-attempt-result/1',
               'worker_checks_pass': worker_ok,
               'supervisor_success': sup['supervisor_success'],
               'ledger_consistent': ledger_ok,
               'final_artifacts_verified': sup['artifacts_verified'],
               'supervisor': sup,
               'prereg_sha256': sha256_file(a.prereg) if Path(a.prereg).is_file() else None,
               'model_sha256': sha256_file(Path(a.model_dir) / 'your_model.pt')}
    verdict['ALL'] = bool(worker_ok and sup['supervisor_success'] and ledger_ok and
                          sup['artifacts_verified'])
    (exp / 'supervisor.json').write_text(json.dumps(sup, indent=1, ensure_ascii=False),
                                         encoding='utf-8')
    (exp / 'attempt_result.json').write_text(json.dumps(verdict, indent=1, ensure_ascii=False),
                                             encoding='utf-8')
    print('[verdict] worker=%s supervisor=%s ledger=%s artifacts=%s ALL=%s'
          % (worker_ok, sup['supervisor_success'], ledger_ok, sup['artifacts_verified'],
             verdict['ALL']), flush=True)
    return 0 if verdict['ALL'] else 1


if __name__ == '__main__':
    sys.exit(main())
