# -*- coding: utf-8 -*-
"""通用候选检查（OPT-04 等）：与 OPT-01 基准数组逐样本比较 + 同输入重复 + 新进程重载。

用法: python opt_checks.py <base_npy_dir> <variant_model_dir> <ledger> <out_dir> [--child <out_dir>]
"""
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import subprocess  # noqa: E402
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

SAMPLES = ['c1_512', 'c7_512', 'c25_512', 'case1_full', 'case7_full']


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def load_case(name):
    if name in SB.BOX:
        return SB.load_input(name)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    src = SB.SRC / ('%s.jpg' % name.split('_')[0])
    arr = np.asarray(Image.open(src).convert('RGB'))
    x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x * 2.0 - 1.0


def u8(a, b):
    d = pixel_io.u8_delta(pixel_io.to_u8(a), pixel_io.to_u8(b))
    diff = np.abs(pixel_io.to_u8(a).astype(np.int16) - pixel_io.to_u8(b).astype(np.int16))
    return {'u8_max': int(d['max_channel_diff']), 'u8_frac': float(d['frac_pixels_any_channel_diff']),
            'u8_identical': bool(d['identical']), 'u8_n_ge2': int((diff >= 2).sum()),
            'u8_n_ge4': int((diff >= 4).sum()), 'u8_n_ge8': int((diff >= 8).sum())}


def child(model_dir, out_dir):
    frozen_mode.apply_frozen_mode()
    out_dir = Path(out_dir)
    r = SB.load_runner(model_dir)
    res = {}
    for name in ('c1_512', 'case1_full'):
        x = load_case(name)
        with torch.no_grad():
            y = r.infer(x, prompt='')
        p = out_dir / ('child_%s.npy' % name)
        np.save(p, y.detach().float().cpu().numpy())
        res[name] = {'npy_sha256': sha(p)}
    (out_dir / 'child_result.json').write_text(json.dumps(res), encoding='utf-8')
    print(json.dumps(res))
    return 0


def main():
    if '--child' in sys.argv:
        i = sys.argv.index('--child')
        return child(Path(sys.argv[i + 1]).resolve(), Path(sys.argv[i + 2]).resolve())
    base_npy = Path(sys.argv[1]).resolve()
    mdir = Path(sys.argv[2]).resolve()
    ledger = Path(sys.argv[3]).resolve()
    out_dir = Path(sys.argv[4]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_mode.apply_frozen_mode()
    led = r_ledger.Ledger.from_file(ledger)
    out = {'schema': 'OPT-checks/1', 'variant_model_dir': str(mdir),
           'runner_sha256': sha(mdir / 'runner.py'),
           'model_sha256': sha(mdir / 'your_model.pt'),
           'base_npy_dir': str(base_npy), 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    status, detail = 'FAIL', ''
    led.acquire_lock('OPT checks')
    try:
        led.open_attempt('opt_checks_%d' % os.getpid(), out_dir, source_sha256=sha(Path(__file__)))
        t0 = time.perf_counter()
        r = SB.load_runner(mdir)
        out['construct_s'] = round(time.perf_counter() - t0, 3)
        out['prewarmed'] = bool(getattr(r, 'prewarmed', None))
        for name in SAMPLES:
            x = load_case(name)
            is4k = name.endswith('_full')
            led.reserve('top', 'optchk_%s' % name, is_4k=is4k)
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                y = r.infer(x, prompt='').detach().float().cpu()
            dt = (time.perf_counter() - t) * 1000.0
            if not bool(torch.isfinite(y).all()):
                raise RuntimeError('%s 非有限' % name)
            base = torch.from_numpy(np.load(base_npy / ('%s_candidate.npy' % name),
                                            allow_pickle=False)).float()
            d = (base - y).abs()
            out.setdefault('samples', {})[name] = {
                'wall_ms': round(dt, 2), 'finite': True, 'range': [float(y.min()), float(y.max())],
                'identical': bool(torch.equal(base, y)), 'float_max_abs': float(d.max()),
                'float_mean_abs': float(d.mean()), **u8(base, y),
                'npy_sha256': None}
            p = out_dir / ('%s_variant.npy' % name)
            np.save(p, y.numpy())
            out['samples'][name]['npy_sha256'] = sha(p)
            print('[chk] %-11s %.0fms identical=%s max=%.3e mean=%.3e u8max=%d frac=%.4f'
                  % (name, dt, out['samples'][name]['identical'], float(d.max()), float(d.mean()),
                     out['samples'][name]['u8_max'], out['samples'][name]['u8_frac']), flush=True)
        x = load_case('c1_512')
        reps = []
        for i in (1, 2):
            led.reserve('top', 'optchk_repeat_%d' % i)
            with torch.no_grad():
                reps.append(r.infer(x, prompt='').detach().float().cpu())
        out['repeat_identical'] = bool(torch.equal(reps[0], reps[1]))
        out['repeat_max_abs'] = float((reps[0] - reps[1]).abs().max())
        np.save(out_dir / 'c1_512_repeat.npy', reps[0].numpy())
        led.reserve('top', 'optchk_child_c1_512')
        led.reserve('top', 'optchk_child_case1_full', is_4k=True)
        cp = subprocess.run([sys.executable, '-B', str(Path(__file__)), str(base_npy), str(mdir),
                             str(ledger), str(out_dir), '--child', str(mdir), str(out_dir)],
                            capture_output=True, text=True, timeout=2400, env=dict(os.environ),
                            encoding='utf-8', errors='replace')
        out['child_rc'] = cp.returncode
        cj = out_dir / 'child_result.json'
        if cp.returncode != 0 or not cj.is_file():
            raise RuntimeError('子进程重载失败 rc=%s %s' % (cp.returncode, (cp.stdout or '')[-200:]))
        cdata = json.loads(cj.read_text(encoding='utf-8'))
        out['reload_identical'] = {
            'c1_512': cdata['c1_512']['npy_sha256'] == out['samples']['c1_512']['npy_sha256'],
            'case1_full': cdata['case1_full']['npy_sha256'] == out['samples']['case1_full']['npy_sha256']}
        if not all(out['reload_identical'].values()):
            raise RuntimeError('新进程重载不一致：%s' % out['reload_identical'])
        out['status'] = 'OK'
        status = 'OK'
    except Exception as e:
        detail = '%s: %s' % (type(e).__name__, e)
        out['error'] = detail
        print('[chk] FAIL %s' % detail, flush=True)
    finally:
        try:
            out['budget_after'] = led._read()['used']
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
        (out_dir / 'opt_checks.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                 encoding='utf-8')
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
