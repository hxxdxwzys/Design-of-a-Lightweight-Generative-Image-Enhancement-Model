# -*- coding: utf-8 -*-
"""OPT-03：冷启动结构测量 + 预热（pre-warm）runner 变体。

H2 假设：官方 512 测速可能包含冷启动效应；在 Runner.__init__ 内做一次 dummy 512 前向
可把「首次被计时 infer」从 ~1.1 s 降到 ~0.33 s，且**数值完全不变**（图像逐字节相同），
代价是加载期多一次前向（~0.3~1.1 s）。

用法: python opt03_cold.py <side: opt01|variant> <run_dir>
"""
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_mode  # noqa: E402
import r_ledger  # noqa: E402
import score01_bench as SB  # noqa: E402


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    side = sys.argv[1]
    run = Path(sys.argv[2]).resolve()
    frozen_mode.apply_frozen_mode()
    mdir = (Path(r'D:\models\opt01\model_dir') if side == 'opt01'
            else run / 'variant' / 'model_dir')
    led = r_ledger.Ledger.from_file(run / 'ledger_opt03.json')
    led.acquire_lock('OPT-03 cold %s' % side)
    rec = {'schema': 'OPT03-cold/1', 'side': side, 'model_dir': str(mdir),
           'runner_sha256': sha(mdir / 'runner.py'), 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    status, detail = 'FAIL', ''
    try:
        led.open_attempt('opt03_cold_%s_%d' % (side, os.getpid()), run,
                         source_sha256=sha(Path(__file__)))
        x = SB.load_input('c1_512')
        t0 = time.perf_counter()
        runner = SB.load_runner(mdir)          # 含变体的预热前向
        torch.cuda.synchronize()
        rec['construct_s'] = round(time.perf_counter() - t0, 3)
        calls = []
        for i in range(6):
            led.reserve('top', 'opt03_%s_call%d' % (side, i + 1))
            torch.cuda.synchronize()
            t = time.perf_counter()
            y = runner.infer(x, prompt='')
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000.0
            if not bool(torch.isfinite(y).all()):
                raise RuntimeError('非有限输出')
            calls.append(round(ms, 2))
        rec['first_calls_ms'] = calls
        rec['steady_median_last4_ms'] = round(float(sorted(calls[2:])[1]), 2)
        p = run / 'npy' / ('cold_%s_out.npy' % side)
        import numpy as np
        np.save(p, y.detach().float().cpu().numpy())
        rec['out_sha256'] = sha(p)
        rec['status'] = 'OK'
        status = 'OK'
        print('[cold %s] construct=%.2fs calls=%s out_sha=%s' % (side, rec['construct_s'], calls,
                                                                 rec['out_sha256'][:16]), flush=True)
    except Exception as e:
        detail = '%s: %s' % (type(e).__name__, e)
        rec['error'] = detail
        print('[cold %s] FAIL %s' % (side, detail), flush=True)
    finally:
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
        (run / 'evidence' / ('cold_%s.json' % side)).write_text(
            json.dumps(rec, indent=1, ensure_ascii=False), encoding='utf-8')
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
