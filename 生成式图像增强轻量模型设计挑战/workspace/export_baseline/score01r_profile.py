# -*- coding: utf-8 -*-
"""SCORE-01R 分段 profiling（独立运行、仅用于解释；§0H 16/18）。

与 SCORE-01 版的区别：**每次调用先扣后跑**（含预热），并在冻结开关下运行。
用法: python score01r_profile.py <model_dir> <ledger> <out_dir> [--tag baseline|candidate]
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_mode  # noqa: E402
import r_ledger  # noqa: E402
import score01_bench as SB  # noqa: E402
import score01r_protocol as P  # noqa: E402


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir')
    ap.add_argument('ledger')
    ap.add_argument('out_dir')
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--inputs', default=','.join(P.INPUTS))
    ap.add_argument('--warmup', type=int, default=1)
    a = ap.parse_args()
    inputs = [s.strip() for s in a.inputs.split(',') if s.strip()]
    need = len(inputs) * (a.warmup + 1)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = frozen_mode.apply_frozen_mode()
    led = r_ledger.Ledger.from_file(a.ledger)
    rec = {'schema': 'SCORE01R-profile/1', 'tag': a.tag, 'flags': flags,
           'model_dir': a.model_dir, 'need_calls': need,
           'model_sha256': sha256_file(Path(a.model_dir) / 'your_model.pt'),
           'runner_sha256': sha256_file(Path(a.model_dir) / 'runner.py'),
           'source_sha256': sha256_file(Path(__file__)), 'per_input': {},
           'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    status, detail = 'FAIL', ''
    led.acquire_lock('SCORE-01R profile %s' % a.tag)
    try:
        d = led._read()
        rem = int(led.limits['top']) - int(d['used']['top'])
        rec['budget_before'] = rem
        if rem < need:
            rec['status'] = 'BUDGET_INSUFFICIENT'
            print('[profile] 额度不足 need=%d rem=%d（0 次调用）' % (need, rem), flush=True)
            return 3
        led.open_attempt('score01r_profile_%s_%d' % (a.tag, os.getpid()), out_dir,
                         source_sha256=rec['source_sha256'], model_sha256=rec['model_sha256'])
        counter = {'n': 0}
        real_sync = torch.cuda.synchronize

        def counting_sync(*args, **kw):
            counter['n'] += 1
            return real_sync(*args, **kw)
        torch.cuda.synchronize = counting_sync
        runner = SB.load_runner(a.model_dir)
        for name in inputs:
            x = SB.load_input(name)
            for i in range(a.warmup):
                led.reserve('top', 'rpro_%s_%s_warm%d' % (a.tag, name, i + 1))
                runner.infer(x, prompt='', timing=None)
            led.reserve('top', 'rpro_%s_%s_timed' % (a.tag, name))
            counter['n'] = 0
            torch.cuda.synchronize()
            timing = {}
            t0 = time.perf_counter()
            y = runner.infer(x, prompt='', timing=timing)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000.0
            rec['per_input'][name] = {
                'timing': timing, 'wall_ms': round(wall, 3),
                'n_cuda_sync': counter['n'],
                'finite': bool(torch.isfinite(y).all()), 'out_shape': list(y.shape)}
            print('[profile %s] %-8s wall=%.1fms syncs=%d %s'
                  % (a.tag, name, wall, counter['n'],
                     {k: round(v * 1000, 2) for k, v in timing.items()}), flush=True)
        torch.cuda.synchronize = real_sync
        rec['status'] = 'OK'
        status = 'OK'
    except Exception as e:
        detail = '%s: %s' % (type(e).__name__, e)
        rec['error'] = detail
        print('[profile] FAIL %s' % detail, flush=True)
    finally:
        rec['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
        (out_dir / ('score01r_profile_%s.json' % a.tag)).write_text(
            json.dumps(rec, indent=1, ensure_ascii=False), encoding='utf-8')
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
