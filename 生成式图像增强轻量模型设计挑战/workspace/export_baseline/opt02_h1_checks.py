# -*- coding: utf-8 -*-
"""OPT-02 H1 复核：变体输出与 OPT-01 存档数组的差异 + 可重复性 + 新进程重载。

用法: python opt02_h1_checks.py <opt01_run> <opt02_run>
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
            'u8_n_ge4': int((diff >= 4).sum())}


def child(run):
    frozen_mode.apply_frozen_mode()
    r = SB.load_runner(run / 'variant' / 'model_dir')
    res = {}
    for name in ('c1_512', 'case1_full'):
        x = load_case(name)
        with torch.no_grad():
            y = r.infer(x, prompt='')
        p = run / 'npy' / ('%s_variant_child.npy' % name)
        np.save(p, y.detach().float().cpu().numpy())
        res[name] = {'npy_sha256': sha(p)}
    (run / 'npy' / 'variant_child.json').write_text(json.dumps(res), encoding='utf-8')
    print(json.dumps(res))
    return 0


def main():
    if '--child' in sys.argv:
        return child(Path(sys.argv[1]).resolve())
    o1 = Path(sys.argv[1]).resolve()
    o2 = Path(sys.argv[2]).resolve()
    frozen_mode.apply_frozen_mode()
    npy = o2 / 'npy'
    npy.mkdir(parents=True, exist_ok=True)
    led = r_ledger.Ledger.from_file(o2 / 'ledger_opt02.json')
    out = {'schema': 'OPT02-h1-checks/1', 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    status, detail = 'FAIL', ''
    led.acquire_lock('OPT-02 H1 checks')
    try:
        led.open_attempt('opt02_h1_checks_%d' % os.getpid(), o2, source_sha256=sha(Path(__file__)))
        r = SB.load_runner(o2 / 'variant' / 'model_dir')
        for name in ('c1_512', 'c7_512', 'c25_512', 'case1_full', 'case7_full'):
            x = load_case(name)
            is4k = name.endswith('_full')
            led.reserve('top', 'opt02_delta_%s' % name, is_4k=is4k)
            with torch.no_grad():
                y = r.infer(x, prompt='').detach().float().cpu()
            if not bool(torch.isfinite(y).all()):
                raise RuntimeError('%s 非有限' % name)
            base = torch.from_numpy(np.load(o1 / 'npy' / ('%s_candidate.npy' % name),
                                            allow_pickle=False)).float()
            d = (base - y).abs()
            rec = {'float_max_abs': float(d.max()), 'float_mean_abs': float(d.mean()),
                   'identical': bool(torch.equal(base, y)), **u8(base, y)}
            out.setdefault('vs_opt01', {})[name] = rec
            print('[h1] %-10s identical=%s float max=%.3e mean=%.3e u8max=%d frac=%.6f'
                  % (name, rec['identical'], rec['float_max_abs'], rec['float_mean_abs'],
                     rec['u8_max'], rec['u8_frac']), flush=True)
        # 同输入重复
        x = load_case('c1_512')
        reps = []
        for i in (1, 2):
            led.reserve('top', 'opt02_repeat_%d' % i)
            with torch.no_grad():
                reps.append(r.infer(x, prompt='').detach().float().cpu())
        out['repeat_identical'] = bool(torch.equal(reps[0], reps[1]))
        out['repeat_max_abs'] = float((reps[0] - reps[1]).abs().max())
        np.save(npy / 'c1_512_variant.npy', reps[0].numpy())
        # 新进程重载：先扣后跑（父进程为子进程两次调用预留）
        led.reserve('top', 'opt02_child_c1_512')
        led.reserve('top', 'opt02_child_case1_full', is_4k=True)
        cp = subprocess.run([sys.executable, '-B', str(Path(__file__)), str(o1), str(o2), '--child'],
                            capture_output=True, text=True, timeout=1800, env=dict(os.environ),
                            encoding='utf-8', errors='replace')
        out['child_rc'] = cp.returncode
        cj = npy / 'variant_child.json'
        if cp.returncode != 0 or not cj.is_file():
            raise RuntimeError('新进程重载失败 rc=%s %s' % (cp.returncode, (cp.stdout or '')[-200:]))
        cdata = json.loads(cj.read_text(encoding='utf-8'))
        saved = {'c1_512': sha(npy / 'c1_512_variant.npy')}
        x4 = load_case('case1_full')
        led.reserve('top', 'opt02_parent_case1_full', is_4k=True)
        with torch.no_grad():
            y4 = r.infer(x4, prompt='').detach().float().cpu()
        np.save(npy / 'case1_full_variant.npy', y4.numpy())
        saved['case1_full'] = sha(npy / 'case1_full_variant.npy')
        out['reload_identical'] = {k: (cdata[k]['npy_sha256'] == saved[k]) for k in saved}
        if not all(out['reload_identical'].values()):
            raise RuntimeError('新进程重载不一致：%s' % out['reload_identical'])
        out['status'] = 'OK'
        status = 'OK'
    except Exception as e:
        detail = '%s: %s' % (type(e).__name__, e)
        out['error'] = detail
        print('[h1] FAIL %s' % detail, flush=True)
    finally:
        try:
            out['budget_after'] = led._read()['used']
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
        (o2 / 'evidence' / 'opt02_h1_checks.json').write_text(
            json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
