# -*- coding: utf-8 -*-
"""STAT-01 输出变化幅度：两侧各跑一次三个 512 裁切，落盘 NPY 并计算 float / uint8 差异。

用法: python stat01_outputs.py <run_dir>
说明：只读复用两侧已冻结导出件；冻结模式；先扣后跑；差异幅度**不代表质量好坏**。
"""
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_mode  # noqa: E402
import pixel_io  # noqa: E402
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
    run = Path(sys.argv[1]).resolve()
    ev = run / 'evidence'
    npy = run / 'npy'
    npy.mkdir(parents=True, exist_ok=True)
    flags = frozen_mode.apply_frozen_mode()
    led = r_ledger.Ledger.from_file(run / 'ledger_stat01.json')
    out = {'schema': 'STAT01-output-delta/1', 'flags': flags,
           'started_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'per_input': {}}
    status, detail = 'FAIL', ''
    led.acquire_lock('STAT-01 outputs')
    try:
        need = 2 * len(P.INPUTS)
        d = led._read()
        rem = int(led.limits['top']) - int(d['used']['top'])
        out['budget_before'] = rem
        if rem < need:
            out['status'] = 'BUDGET_INSUFFICIENT'
            print('[stat01] 额度不足 need=%d rem=%d（0 次调用）' % (need, rem), flush=True)
            return 3
        led.open_attempt('stat01_outputs_%d' % os.getpid(), run,
                         source_sha256=sha256_file(Path(__file__)))
        runners = {'A': SB.load_runner(run / 'mixed' / 'model_dir'),
                   'B': SB.load_runner(run / 'allfp16' / 'model_dir')}
        res = {}
        for name in P.INPUTS:
            x = SB.load_input(name)
            ys = {}
            for lab in ('A', 'B'):
                led.reserve('top', 'stat01_out_%s_%s' % (lab, name))
                with torch.no_grad():
                    y = runners[lab].infer(x, prompt='')
                p = npy / ('%s_%s.npy' % (name, lab))
                np.save(p, y.detach().float().cpu().numpy())
                ys[lab] = y.detach().float().cpu()
                if not bool(torch.isfinite(ys[lab]).all()):
                    raise RuntimeError('%s/%s 非有限输出' % (lab, name))
            a, b = ys['A'], ys['B']
            df = (a - b).abs()
            ua, ub = pixel_io.to_u8(a), pixel_io.to_u8(b)
            delta = pixel_io.u8_delta(ua, ub)
            diff = np.abs(ua.astype(np.int16) - ub.astype(np.int16))
            hist = {str(k): int((diff == k).sum()) for k in range(4)}
            hist['>=4'] = int((diff >= 4).sum())
            res[name] = {'float_max_abs': float(df.max()), 'float_mean_abs': float(df.mean()),
                         'u8_max_channel_diff': int(delta['max_channel_diff']),
                         'u8_frac_pixels_any_channel_diff': float(
                             delta['frac_pixels_any_channel_diff']),
                         'u8_identical': bool(delta['identical']),
                         'u8_abs_histogram': hist,
                         'A_range': [float(a.min()), float(a.max())],
                         'B_range': [float(b.min()), float(b.max())],
                         'A_npy_sha256': sha256_file(npy / ('%s_A.npy' % name)),
                         'B_npy_sha256': sha256_file(npy / ('%s_B.npy' % name))}
            print('[stat01] %-8s float max=%.4e mean=%.4e | u8 maxdiff=%d frac=%.6f identical=%s'
                  % (name, res[name]['float_max_abs'], res[name]['float_mean_abs'],
                     res[name]['u8_max_channel_diff'],
                     res[name]['u8_frac_pixels_any_channel_diff'],
                     res[name]['u8_identical']), flush=True)
        out['per_input'] = res
        out['status'] = 'OK'
        status = 'OK'
    except Exception as e:
        detail = '%s: %s' % (type(e).__name__, e)
        out['error'] = detail
        print('[stat01] FAIL %s' % detail, flush=True)
    finally:
        try:
            out['budget_after'] = led._read()['used']
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
        (ev / 'stat01_output_delta.json').write_text(
            json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
