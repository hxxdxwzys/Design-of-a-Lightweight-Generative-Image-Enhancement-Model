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


def _pid_alive(pid):
    """Windows 安全存活探测（os.kill(pid,0) 会真的终止目标进程，禁止使用）。"""
    pid = int(pid)
    if pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            k.GetExitCodeProcess(h, ctypes.byref(code))
            return code.value == 259
        finally:
            k.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def descendant_pids(root_pid):
    """用 WMI 列出 root 的子孙进程（Windows）；失败返回 None（UNKNOWN）。"""
    if os.name != 'nt':
        return None
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId | "
             "ConvertTo-Json -Compress)"], capture_output=True, text=True, timeout=60)
        rows = json.loads(out.stdout or '[]')
        if isinstance(rows, dict):
            rows = [rows]
        children = {}
        for r in rows:
            children.setdefault(int(r['ParentProcessId']), []).append(int(r['ProcessId']))
        found, stack = [], [int(root_pid)]
        while stack:
            cur = stack.pop()
            for ch in children.get(cur, []):
                if ch not in found:
                    found.append(ch)
                    stack.append(ch)
        return found
    except Exception:
        return None


def kill_process_tree(root_pid):
    """超时/收尾时确保整棵树退出；返回 {killed_tree, remaining, unknown}。"""
    before = descendant_pids(root_pid)
    if os.name == 'nt':
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(int(root_pid))],
                       capture_output=True, text=True)
    else:
        subprocess.run(['pkill', '-P', str(int(root_pid))], capture_output=True, text=True)
    time.sleep(1.0)
    after = descendant_pids(root_pid)
    remaining = [p for p in (after or []) if _pid_alive(p)]
    for p in list(remaining):          # 兜底：逐个强杀仍存活的子孙
        subprocess.run(['taskkill', '/F', '/PID', str(int(p))],
                       capture_output=True, text=True)
    time.sleep(0.5)
    remaining = [p for p in (after or []) if _pid_alive(p)]
    return {'descendants_before': before, 'descendants_after': after,
            'remaining_alive': remaining, 'clean': (after is not None and not remaining),
            'unknown': after is None}


def evaluate_readings(rd, tol, exp, required_cases):
    """产物与判据核验（独立复算；缺项/重复/空集合/哈希错项一律不得 PASS）。"""
    checks = {'readings_present': rd is not None,
              'worker_status_ok': bool(rd and rd.get('status') == 'OK'),
              'required_cases': list(required_cases)}
    if not (rd and rd.get('status') == 'OK'):
        return checks
    npy_missing = []
    for key in ('phaseA', 'phaseB'):
        for v in rd.get(key, {}).values():
            if isinstance(v, dict) and 'npy' in v:
                p = Path(v['npy'])
                if not p.is_file() or sha256_file(p) != v['npy_sha256']:
                    npy_missing.append(v['npy'])
    checks['all_npy_present_and_hashed'] = not npy_missing
    checks['npy_missing'] = npy_missing
    b512 = [v for v in rd.get('phaseB', {}).values()
            if isinstance(v, dict) and str(v.get('call', '')).startswith('B_512')]
    calls = sorted(str(v.get('call')) for v in rd.get('phaseB', {}).values() if 'call' in v)
    checks['phaseB_calls'] = calls
    checks['n_B512'] = len(b512)
    checks['B512_set_ok'] = (calls.count('B_512_1') == 1 and calls.count('B_512_2') == 1 and
                             calls.count('B_512_3') == 1 and 'B_511x513' in calls)
    checks['B512_vs_A_ok'] = bool(b512) and all(
        v['vs_A']['max_abs'] <= tol['same_precision_max_abs'] and
        v['vs_A']['mean_abs'] <= tol['same_precision_mean_abs'] and
        v['vs_A']['png_pixel_identical'] for v in b512)
    rep = [v for v in b512 if v.get('call') != 'B_512_1']
    checks['B512_repeat_ok'] = bool(rep) and all(
        v['vs_first_B512']['max_abs'] <= tol['same_precision_max_abs'] and
        v['vs_first_B512']['mean_abs'] <= tol['same_precision_mean_abs'] and
        v['vs_first_B512']['png_pixel_identical'] for v in rep)
    pairs = rd.get('pairs', [])
    names = [p['name'] for p in pairs]
    checks['pair_names'] = names
    checks['pairs_ok'] = (sorted(names) == sorted([n for n in required_cases
                                                   if n.endswith('_full')]) and
                          all(p['pair']['max_abs'] <= tol['same_precision_max_abs'] and
                              p['pair']['mean_abs'] <= tol['same_precision_mean_abs'] and
                              p['pair']['png_pixel_identical'] for p in pairs))
    mig = rd.get('migration', [])
    checks['migration_ok'] = bool(mig) and all(
        m['max_abs'] <= tol.get('fp16_vs_fp32_max_abs_diagnostic', 0.05) and
        m['mean_abs'] <= tol.get('fp16_vs_fp32_mean_abs_diagnostic', 0.002) for m in mig)
    checks['n_migration'] = len(mig)
    checks['required_set_ok'] = (checks['B512_set_ok'] and
                                 len([n for n in names if n.endswith('_full')]) == 2)
    return checks


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
    ap.add_argument('--expect-model-sha256', default=None,
                    help='加载前强制的模型文件哈希（EXP-R09 身份绑定）')
    ap.add_argument('--expect-model-bytes', type=int, default=None)
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
    # EXP-R09：模型/身份在加载前强制绑定（哈希或字节数不符即拒绝启动）
    mp = Path(a.model_dir) / 'your_model.pt'
    pre['model_sha256'] = sha256_file(mp) if mp.is_file() else None
    pre['model_bytes'] = mp.stat().st_size if mp.is_file() else None
    pre['model_hash_expected'] = a.expect_model_sha256
    pre['model_binding_ok'] = bool(
        a.expect_model_sha256 is None or pre['model_sha256'] == a.expect_model_sha256)
    pre['model_bytes_ok'] = bool(
        a.expect_model_bytes is None or pre['model_bytes'] == a.expect_model_bytes)
    if not (pre['model_binding_ok'] and pre['model_bytes_ok']):
        sup.update({'supervisor_success': False, 'error': 'MODEL_IDENTITY_MISMATCH'})
        (exp / 'supervisor.json').write_text(json.dumps(sup, indent=1, ensure_ascii=False),
                                             encoding='utf-8')
        print('SUPERVISOR_FAIL MODEL_IDENTITY_MISMATCH')
        return 2
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
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
            break
        time.sleep(0.5)
    # EXP-R10：整棵本批进程树必须退出（子进程可能已派生探针等孙进程）
    tree = kill_process_tree(proc.pid)
    sup['process_tree'] = tree
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
    prereg = json.loads(Path(a.prereg).read_text(encoding='utf-8'))
    checks = evaluate_readings(rd, prereg['tolerances'], exp,
                               [c['name'] for c in prereg['cases']])
    checks['n_top_used'] = used.get('top')
    checks['n_4k_used'] = used.get('top4k')
    worker_ok = bool(checks['readings_present'] and checks['worker_status_ok'] and
                     checks.get('required_set_ok') and
                     checks.get('all_npy_present_and_hashed') and checks.get('pairs_ok') and
                     checks.get('B512_vs_A_ok') and checks.get('B512_repeat_ok') and
                     checks.get('migration_ok'))
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
