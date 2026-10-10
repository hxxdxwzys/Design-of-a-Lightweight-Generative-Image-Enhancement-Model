# -*- coding: utf-8 -*-
"""OPT-01 验证：候选自身可重复（同输入/新进程）+ sanity + 与 A2 及 fp32 参考的差异幅度。

用法: python opt01_verify.py <run_dir> [--child]
不含跨精度逐位等价判据（D34：跨精度路线不追求逐位等价）；判据为可重复性+sanity+差异如实测量。
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
import r_verify  # noqa: E402
import score01_bench as SB  # noqa: E402

A2_RUN = Path(r'C:\models\exports\DELIV02_20260918_073810')
CASES = {'case1_crop511x513': (1792, 1280, 2305, 1791), 'case1_full': None, 'case7_full': None}
SRC = SB.SRC


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def load_case(name):
    if name in SB.BOX:
        return SB.load_input(name)
    fn, box = SRC / ('%s.jpg' % name.split('_')[0]), CASES[name]
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(fn).convert('RGB')
    if box:
        im = im.crop(box)
    arr = np.asarray(im)
    x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x * 2.0 - 1.0


def u8_stats(a, b):
    ua, ub = pixel_io.to_u8(a), pixel_io.to_u8(b)
    d = pixel_io.u8_delta(ua, ub)
    diff = np.abs(ua.astype(np.int16) - ub.astype(np.int16))
    return {'u8_max_channel_diff': int(d['max_channel_diff']),
            'u8_frac_pixels_any_channel_diff': float(d['frac_pixels_any_channel_diff']),
            'u8_identical': bool(d['identical']),
            'u8_n_ge2': int((diff >= 2).sum()), 'u8_n_ge4': int((diff >= 4).sum())}


def child_main(run):
    """全新进程：只做推理并落盘（不碰锁/账本——父进程已按「先扣后跑」为这两次调用预留）。"""
    frozen_mode.apply_frozen_mode()
    res = {}
    status = 'FAIL'
    try:
        r = SB.load_runner(run / 'model_dir')
        for name in ('c1_512', 'case1_full'):
            x = load_case(name)
            with torch.no_grad():
                y = r.infer(x, prompt='')
            p = run / 'npy' / ('%s_candidate_child.npy' % name)
            np.save(p, y.detach().float().cpu().numpy())
            res[name] = {'npy': str(p), 'npy_sha256': sha(p),
                         'finite': bool(torch.isfinite(y).all()),
                         'range': [float(y.min()), float(y.max())]}
        (run / 'npy' / 'child_result.json').write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                       encoding='utf-8')
        status = 'OK'
    except Exception as e:
        res['error'] = '%s: %s' % (type(e).__name__, e)
    print(json.dumps(res, ensure_ascii=False))
    return 0 if status == 'OK' else 1


def reload_main(run):
    """仅补做「新进程重载」比对：父进程预留两次调用（先扣后跑）后再派生全新进程。"""
    frozen_mode.apply_frozen_mode()
    ev = run / 'evidence'
    npy = run / 'npy'
    led = r_ledger.Ledger.from_file(run / 'ledger_opt01.json')
    led.acquire_lock('OPT-01 reload check')
    status, detail = 'FAIL', ''
    try:
        led.open_attempt('opt01_reload_%d' % os.getpid(), run,
                         source_sha256=sha(Path(__file__)))
        led.reserve('top', 'opt01_child_c1_512')
        led.reserve('top', 'opt01_child_case1_full', is_4k=True)
        cp = subprocess.run([sys.executable, '-B', str(Path(__file__)), str(run), '--child'],
                            capture_output=True, text=True, timeout=1800,
                            env=dict(os.environ), encoding='utf-8', errors='replace')
        out = json.loads((ev / 'opt01_verify.json').read_text(encoding='utf-8'))
        out['child_rc'] = cp.returncode
        out['child_tail'] = ((cp.stdout or '') + (cp.stderr or '')).strip()[-300:]
        cr = npy / 'child_result.json'
        if cp.returncode != 0 or not cr.is_file():
            raise RuntimeError('新进程重载失败：rc=%s %s' % (cp.returncode, out['child_tail']))
        child = json.loads(cr.read_text(encoding='utf-8'))
        for name in ('c1_512', 'case1_full'):
            parent_sha = out['checks'][name]['npy_sha256']
            same = child[name]['npy_sha256'] == parent_sha
            out['checks'][name]['new_process_identical'] = bool(same)
            out['checks'][name]['child_npy_sha256'] = child[name]['npy_sha256']
            if not same:
                raise RuntimeError('新进程重载与本次不一致：%s' % name)
        out['status'] = 'OK'
        status = 'OK'
        (ev / 'opt01_verify.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                              encoding='utf-8')
        print('[reload] OK 新进程与本次逐位一致（c1_512 + case1_full）', flush=True)
    except Exception as e:
        detail = '%s: %s' % (type(e).__name__, e)
        print('[reload] FAIL %s' % detail, flush=True)
    finally:
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
    return 0 if status == 'OK' else 1


def main():
    run = Path(sys.argv[1]).resolve()
    if '--child' in sys.argv:
        return child_main(run)
    if '--reload-only' in sys.argv:
        return reload_main(run)
    flags = frozen_mode.apply_frozen_mode()
    ev = run / 'evidence'
    npy = run / 'npy'
    npy.mkdir(parents=True, exist_ok=True)
    led = r_ledger.Ledger.from_file(run / 'ledger_opt01.json')
    out = {'schema': 'OPT01-verify/1', 'flags': flags, 'checks': {}, 'deltas': {},
           'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    status, detail = 'FAIL', ''
    led.acquire_lock('OPT-01 verify')
    try:
        led.open_attempt('opt01_verify_%d' % os.getpid(), run, source_sha256=sha(Path(__file__)))
        cand = SB.load_runner(run / 'model_dir')
        outs = {}
        # ── 512：候选同输入两次（可重复性）+ 与 A2 记录件及 fp32 参考的差异 ──
        for name in ('c1_512', 'c7_512', 'c25_512'):
            x = load_case(name)
            ys = []
            for i in (1, 2):
                led.reserve('top', 'opt01_verify_%s_r%d' % (name, i))
                with torch.no_grad():
                    y = cand.infer(x, prompt='')
                ys.append(y.detach().float().cpu())
                if not bool(torch.isfinite(y).all()):
                    raise RuntimeError('%s 非有限输出' % name)
            rep = float((ys[0] - ys[1]).abs().max())
            outs[name] = ys[0]
            np.save(npy / ('%s_candidate.npy' % name), ys[0].numpy())
            rec = {'repeat_max_abs': rep, 'repeat_identical': bool(torch.equal(ys[0], ys[1])),
                   'finite': True, 'range': [float(ys[0].min()), float(ys[0].max())],
                   'npy_sha256': sha(npy / ('%s_candidate.npy' % name))}
            if name == 'c1_512':
                a2 = torch.from_numpy(np.load(A2_RUN / 'npy' / 'B_512_r1.npy',
                                              allow_pickle=False)).float()
                rec['vs_A2'] = {'float_max_abs': float((a2 - ys[0]).abs().max()),
                                'float_mean_abs': float((a2 - ys[0]).abs().mean()),
                                **u8_stats(a2, ys[0])}
                fp32 = torch.from_numpy(np.load(r_verify.FP32_REF['case1_crop512'],
                                                allow_pickle=False)).float()
                rec['vs_fp32_reference'] = {'float_max_abs': float((fp32 - ys[0]).abs().max()),
                                            'float_mean_abs': float((fp32 - ys[0]).abs().mean()),
                                            **u8_stats(fp32, ys[0])}
            out['checks'][name] = rec
            if rep != 0.0:
                raise RuntimeError('%s 同输入重复非逐位一致：%s' % (name, rep))
            print('[verify] %-10s repeat=%.1e vs_A2=%s' % (
                name, rep, rec.get('vs_A2', {}).get('float_max_abs')), flush=True)
        # ── 4K：case1 / case7 完成且有限；与 A2 记录输出比对 ──
        for name in ('case1_full', 'case7_full'):
            x = load_case(name)
            led.reserve('top', 'opt01_verify_%s' % name, is_4k=True)
            with torch.no_grad():
                y = cand.infer(x, prompt='', timing={})
            y = y.detach().float().cpu()
            if not bool(torch.isfinite(y).all()):
                raise RuntimeError('%s 非有限输出' % name)
            np.save(npy / ('%s_candidate.npy' % name), y.numpy())
            a2 = torch.from_numpy(np.load(A2_RUN / 'npy' / ('%s_B.npy' % name),
                                          allow_pickle=False)).float()
            out['checks'][name] = {
                'finite': True, 'range': [float(y.min()), float(y.max())],
                'npy_sha256': sha(npy / ('%s_candidate.npy' % name)),
                'vs_A2': {'float_max_abs': float((a2 - y).abs().max()),
                          'float_mean_abs': float((a2 - y).abs().mean()), **u8_stats(a2, y)}}
            print('[verify] %-10s 4K ok finite range=[%.3f,%.3f] vs_A2 max=%.4f mean=%.4e'
                  % (name, y.min(), y.max(), out['checks'][name]['vs_A2']['float_max_abs'],
                     out['checks'][name]['vs_A2']['float_mean_abs']), flush=True)
        del cand
        torch.cuda.empty_cache()
        # ── 新进程重载：与本次数组逐位比较 ──
        cp = subprocess.run([sys.executable, '-B', str(Path(__file__)), str(run), '--child'],
                            capture_output=True, text=True, timeout=1800,
                            env=dict(os.environ), encoding='utf-8', errors='replace')
        out['child_rc'] = cp.returncode
        out['child_tail'] = (cp.stdout or '').strip()[-300:]
        cr = npy / 'child_result.json'
        if cp.returncode != 0 or not cr.is_file():
            raise RuntimeError('新进程重载失败：rc=%s %s' % (cp.returncode, out['child_tail']))
        child = json.loads(cr.read_text(encoding='utf-8'))
        for name in ('c1_512', 'case1_full'):
            same = child[name]['npy_sha256'] == out['checks'][name]['npy_sha256']
            out['checks'][name]['new_process_identical'] = bool(same)
            if not same:
                raise RuntimeError('新进程重载与本次不一致：%s' % name)
        out['status'] = 'OK'
        status = 'OK'
    except Exception as e:
        detail = '%s: %s' % (type(e).__name__, e)
        out['error'] = detail
        print('[verify] FAIL %s' % detail, flush=True)
    finally:
        try:
            out['budget_after'] = led._read()['used']
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
        (ev / 'opt01_verify.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                              encoding='utf-8')
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
