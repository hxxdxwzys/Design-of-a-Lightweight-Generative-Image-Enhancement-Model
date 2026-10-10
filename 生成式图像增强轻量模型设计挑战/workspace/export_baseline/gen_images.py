# -*- coding: utf-8 -*-
"""C2：由交付 runner（JIT bundle）生成 100 张 4K 图。

用法: python gen_images.py <run_dir> [--limit N] [--budget-min 150]
产出: <run_dir>/output_dir/caseN.jpg + <run_dir>/evidence/generation_manifest.json
失败即 PARTIAL 并停止（不切候选、不重试、不补图）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
TEST_DIR = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集')
JPEG = {'quality': 98, 'subsampling': 0, 'optimize': False, 'progressive': False}


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def load_runner(model_dir):
    spec = importlib.util.spec_from_file_location('delivery_runner_pkg',
                                                  str(Path(model_dir) / 'runner.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Runner(str(model_dir))


def main():
    run = Path(sys.argv[1]).resolve()
    limit = 100
    budget_min = 150.0
    ledger_path = None
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit') + 1])
    if '--budget-min' in sys.argv:
        budget_min = float(sys.argv[sys.argv.index('--budget-min') + 1])
    if '--ledger' in sys.argv:
        ledger_path = sys.argv[sys.argv.index('--ledger') + 1]
    from disk_guard import require_free
    require_free(run, 8.0, 'C2 出图前水位（100 图 + 证据）')
    out_dir = run / 'output_dir'
    ev = run / 'evidence'
    out_dir.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)
    model_dir = run / 'model_dir'
    runner = load_runner(model_dir)
    manifest = {
        'schema': 'EXPORT01-generation-manifest/1',
        'bundle': {'model_sha256': sha256_file(model_dir / 'your_model.pt'),
                   'model_bytes': (model_dir / 'your_model.pt').stat().st_size,
                   'runner_sha256': sha256_file(model_dir / 'runner.py'),
                   'recipe_sha256': sha256_file(model_dir / 'models_export.json')},
        'jpeg_params': JPEG,
        'input_dir': str(TEST_DIR),
        'budget_minutes': budget_min,
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'images': [],
        'status': 'RUNNING',
    }
    # ── C2 生产调用必须计入唯一持久总账（先扣后跑、失败不退）──
    led = None
    if ledger_path:
        import r_ledger
        led = r_ledger.Ledger.from_file(ledger_path)
        led.acquire_lock('C2 production')
        led.open_attempt('c2_prod_%s' % time.strftime('%Y%m%d_%H%M%S'), run)
        _d = led._read()
        rem = {k: int(led.limits[k]) - int(_d['used'][k]) for k in ('top', 'top4k')}
        wall_left = float(led.limits['wall_s']) - float(_d.get('supervision_used_s') or 0.0)
        manifest['ledger'] = {'path': str(ledger_path), 'remaining_before': rem,
                              'wall_left_s': round(wall_left, 1),
                              'supervision_used_s_before': _d.get('supervision_used_s')}
        if rem['top'] < limit or rem['top4k'] < limit:
            led.release_lock()
            raise SystemExit('C2 出图额度不足：top 余 %d、4K 余 %d，需 %d'
                             % (rem['top'], rem['top4k'], limit))
        print('[gen] ledger charged mode, remaining=%s wall_left=%.0fs'
              % (rem, wall_left), flush=True)

        def _emergency_close():          # 未预期异常/退出也不留未关闭 attempt 与锁
            try:
                if led._read().get('open_attempt'):
                    led.close_attempt('FAIL', note='unexpected exit (atexit)')
                led.release_lock()
            except Exception:
                pass
        import atexit
        atexit.register(_emergency_close)
    t_start = time.perf_counter()
    for i in range(1, limit + 1):
        rec = {'case': i, 'status': 'PENDING'}
        src = TEST_DIR / ('case%d.jpg' % i)
        try:
            if not src.is_file():
                raise FileNotFoundError(str(src))
            elapsed_min = (time.perf_counter() - t_start) / 60.0
            if elapsed_min > budget_min:
                rec.update({'status': 'BUDGET_STOP', 'elapsed_min': round(elapsed_min, 3)})
                manifest['images'].append(rec)
                manifest['status'] = 'PARTIAL'
                break
            im = Image.open(src).convert('RGB')
            arr = np.asarray(im)
            rec['input_wh'] = [int(im.size[0]), int(im.size[1])]
            rec['input_file_sha256'] = sha256_file(src)
            rec['input_pixels_sha256'] = hashlib.sha256(arr.tobytes()).hexdigest()
            x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            x = x * 2.0 - 1.0
            timing = {}
            if led is not None:
                led.reserve('top', 'prod_case%d' % i, is_4k=bool(max(im.size) >= 3000))
            t0 = time.perf_counter()
            out = runner.infer(x, prompt='', timing=timing)
            rec['wall_s'] = round(time.perf_counter() - t0, 3)
            rec['timing'] = timing
            rec['output_shape'] = list(out.shape)
            rec['output_dtype'] = str(out.dtype)
            rec['output_tensor_sha256'] = hashlib.sha256(
                out.detach().cpu().float().numpy().tobytes()).hexdigest()
            import pixel_io          # DELIV-02：唯一 8 位恢复入口（最近色阶，禁止二次截断）
            u8 = pixel_io.to_u8(out)
            rec['grid_residual'] = pixel_io.grid_residual(out)
            dst = out_dir / ('case%d.jpg' % i)
            Image.fromarray(u8).save(dst, format='JPEG', **JPEG)
            rec['output_file'] = dst.name
            rec['output_file_sha256'] = sha256_file(dst)
            dec = np.asarray(Image.open(dst).convert('RGB'))
            rec['output_decoded_pixels_sha256'] = hashlib.sha256(dec.tobytes()).hexdigest()
            rec['output_bytes'] = dst.stat().st_size
            rec['status'] = 'OK'
        except Exception as e:
            rec.update({'status': 'FAILED', 'error': '%s: %s' % (type(e).__name__, e)})
            manifest['images'].append(rec)
            manifest['status'] = 'PARTIAL'
            break
        manifest['images'].append(rec)
        print('[%3d] %s  %.1fs  %s' % (i, rec['status'], rec.get('wall_s', 0.0),
                                       rec.get('output_file', '')), flush=True)

    ok = [r for r in manifest['images'] if r['status'] == 'OK']
    manifest['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    manifest['elapsed_s'] = round(time.perf_counter() - t_start, 3)
    manifest['n_ok'] = len(ok)
    manifest['n_total_requested'] = limit
    manifest['forwards_used'] = len(manifest['images'])
    manifest['output_bytes_total'] = int(sum(r.get('output_bytes', 0) for r in ok))
    if manifest['status'] == 'RUNNING':
        manifest['status'] = 'OK' if len(ok) == limit else 'PARTIAL'
    if led is not None:
        led.note('c2_production_done', {'n_ok': manifest['n_ok'], 'status': manifest['status'],
                                        'elapsed_s': manifest['elapsed_s']})
        manifest['ledger']['summary_after'] = led.summary()['used']
        led.close_attempt(manifest['status'],
                          note='n_ok=%d/%d' % (manifest['n_ok'], limit))
        led.release_lock()
    (ev / 'generation_manifest.json').write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in
                      ('status', 'n_ok', 'forwards_used', 'elapsed_s', 'output_bytes_total')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
