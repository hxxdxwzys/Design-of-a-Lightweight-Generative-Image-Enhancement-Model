# -*- coding: utf-8 -*-
"""SCORE-01 分段 profiling（§0G 第 2/3 步，独立运行，不与无 profiling 成绩混用）。

用法: python score01_profile.py <model_dir> <ledger> <run_dir>
产出: <run>/evidence/profile_baseline.json
内容:
  1) 每输入 2 次预热 + 1 次带 timing 的真实 infer（阶段耗时来自交付 runner 自身 timing 字典）
  2) 每次 infer 的 cuda 同步次数（计数包装，不改变行为）
  3) 纯 CPU 替身测量：同一 runner 编排 + stub 模块（同形状张量），量化 Python/检查开销
  4) 单次 torch.cuda.synchronize() 成本
"""
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r_ledger  # noqa: E402
import score01_bench as B  # noqa: E402


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


class Stub(torch.nn.Module):
    """CPU 替身：只回同形状张量，用于测量编排/检查开销（不代表数值）。"""

    def __init__(self, ref):
        super().__init__()
        self.ref = ref
        self.calls = []

    def encode(self, x, tile_px):
        self.calls.append(('encode', tuple(x.shape), int(tile_px)))
        n = int(tile_px) // 8
        return torch.zeros(x.shape[0], 4, n, n, dtype=torch.float16)

    def features(self, x):
        self.calls.append(('features', tuple(x.shape)))
        return torch.zeros(x.shape[0], 1024, 768, dtype=torch.float16)

    def step(self, lq, z, tc, tn, feats):
        self.calls.append(('step', tuple(lq.shape)))
        return torch.zeros_like(lq)

    def decode(self, z, tile):
        self.calls.append(('decode', tuple(z.shape), int(tile)))
        n = int(tile) * 8
        return torch.zeros(z.shape[0], 3, n, n, dtype=torch.float32)

    def post(self, sr, src):
        self.calls.append(('post', tuple(sr.shape)))
        return torch.zeros_like(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir')
    ap.add_argument('ledger')
    ap.add_argument('run_dir')
    ap.add_argument('--inputs', default='c1_512,c7_512,c25_512')
    a = ap.parse_args()
    run = Path(a.run_dir)
    ev = run / 'evidence'
    ev.mkdir(parents=True, exist_ok=True)
    rec = {'schema': 'SCORE01-profile/1', 'model_dir': a.model_dir,
           'model_sha256': sha256_file(Path(a.model_dir) / 'your_model.pt')}
    led = r_ledger.Ledger.from_file(a.ledger)
    led.acquire_lock('SCORE-01 profile')
    status, detail = 'FAIL', ''
    try:
        led.open_attempt('profile_baseline', run, model_sha256=rec['model_sha256'],
                         source_sha256=sha256_file(HERE / 'score01_profile.py'))
        # 单次同步成本
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(200):
            torch.cuda.synchronize()
        rec['sync_cost_us'] = round((time.perf_counter() - t) / 200 * 1e6, 2)
        # 同步计数包装
        sync_counter = {'n': 0}
        real_sync = torch.cuda.synchronize

        def counting_sync(*args, **kw):
            sync_counter['n'] += 1
            return real_sync(*args, **kw)

        runner = B.load_runner(a.model_dir)
        rec['runner_load_done'] = True
        per_input = {}
        for name in [s.strip() for s in a.inputs.split(',') if s.strip()]:
            x = B.load_input(name)
            entry = {'input_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest()}
            for i in range(2):      # 预热
                led.reserve('top', 'profile_%s_warm%d' % (name, i + 1))
                runner.infer(x, prompt='', timing=None)
            torch.cuda.synchronize()
            sync_counter['n'] = 0
            torch.cuda.synchronize()
            sync_counter['n'] = 0
            torch.cuda.synchronize = counting_sync
            try:
                timing = {}
                t = time.perf_counter()
                y = runner.infer(x, prompt='', timing=timing)
                torch.cuda.synchronize()
                wall = (time.perf_counter() - t) * 1000.0
            finally:
                torch.cuda.synchronize = real_sync
            led.reserve('top', 'profile_%s_timed' % name)
            entry.update({'timing': timing, 'wall_ms_incl_sync': round(wall, 3),
                          'n_cuda_sync_per_infer': sync_counter['n'],
                          'finite': bool(torch.isfinite(y).all()),
                          'out_shape': list(y.shape)})
            per_input[name] = entry
            print('[profile] %-8s wall=%.1fms syncs=%d %s'
                  % (name, wall, entry['n_cuda_sync_per_infer'],
                     {k: round(v * 1000, 1) for k, v in timing.items()}), flush=True)
        rec['per_input'] = per_input
        # 纯 CPU 替身（同一编排，stub 模块）
        stub = Stub(None)
        runner.model = stub
        x = B.load_input('c1_512')
        runner.model = stub
        orig_dev_sync = runner._dev_sync
        cpu_runs = []
        for trial in range(3):
            t = time.perf_counter()
            runner.infer(x, prompt='', timing=None)
            cpu_runs.append(round((time.perf_counter() - t) * 1000.0, 3))
        rec['cpu_stub_orchestration_ms'] = cpu_runs
        rec['cpu_stub_calls'] = [list(c) for c in stub.calls[:8]]
        # 无同步替身（计时为 None 时跳过 _dev_sync 的等价效果，用 CPU 测量）
        runner._dev_sync = lambda: None
        cpu_nosync = []
        for trial in range(3):
            t = time.perf_counter()
            runner.infer(x, prompt='', timing=None)
            cpu_nosync.append(round((time.perf_counter() - t) * 1000.0, 3))
        rec['cpu_stub_orchestration_no_devsync_ms'] = cpu_nosync
        runner._dev_sync = orig_dev_sync
        rec['status'] = 'OK'
        status = 'OK'
    except Exception as e:
        detail = '%s: %s' % (type(e).__name__, e)
        rec['error'] = detail
        print('[profile] FAIL %s' % detail, flush=True)
    finally:
        led.close_attempt(status, note=detail)
        led.release_lock()
        (ev / 'profile_baseline.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                                                  encoding='utf-8')
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
