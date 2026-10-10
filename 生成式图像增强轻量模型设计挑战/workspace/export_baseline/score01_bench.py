# -*- coding: utf-8 -*-
"""SCORE-01 时延基准/候选测量（§0G 第 2 步）：真实交付 Runner 入口，冷启动 + 稳态。

用法: python score01_bench.py <model_dir> <label> <ledger> <out_dir> [--inputs a,b,c]
                             [--warmup 3] [--timed 20] [--attempt ID] [--frozen]
协议：每输入 1 次冷启动（进程内首个调用）+ N 次预热 + M 次计时；调用前后各一次 cuda 同步；
      每次调用先扣账（'top'）；原始读数与统计全部落盘。
"""
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r_ledger  # noqa: E402

SRC = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集')
BOX = {
    "c1_512": ("case1.jpg", (1792, 1280, 2304, 1792)),
    "c7_512": ("case7.jpg", (1792, 1280, 2304, 1792)),
    "c25_512": ("case25.jpg", (1280, 1792, 1792, 2304)),
}


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def load_runner(model_dir):
    spec = importlib.util.spec_from_file_location('bench_runner_pkg',
                                                  str(Path(model_dir) / 'runner.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Runner(str(model_dir))


def load_input(name):
    fn, box = BOX[name]
    im = Image.open(SRC / fn).convert('RGB')
    if box:
        im = im.crop(box)
    arr = np.asarray(im)
    x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x * 2.0 - 1.0


def nvsmi(field):
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=%s' % field,
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir')
    ap.add_argument('label')
    ap.add_argument('ledger')
    ap.add_argument('out_dir')
    ap.add_argument('--inputs', default='c1_512,c7_512,c25_512')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--timed', type=int, default=20)
    ap.add_argument('--attempt', default=None)
    ap.add_argument('--frozen', action='store_true',
                    help='启用冻结执行开关（默认关闭＝贴近平台环境，仅 runner 自带开关生效）')
    a = ap.parse_args()

    flags = {}
    if a.frozen:
        import frozen_mode
        flags = frozen_mode.apply_frozen_mode()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(a.model_dir)
    led = r_ledger.Ledger.from_file(a.ledger)
    attempt = a.attempt or ('bench_%s_%d' % (a.label, os.getpid()))
    led.acquire_lock('SCORE-01 bench %s' % a.label)
    rec = {'schema': 'SCORE01-bench/1', 'label': a.label, 'model_dir': str(model_dir),
           'attempt': attempt, 'pid': os.getpid(),
           'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'frozen_flags': bool(a.frozen),
           'flags': flags, 'warmup': a.warmup, 'timed': a.timed,
           'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu_name': torch.cuda.get_device_name(0),
           'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
           'cudnn_benchmark': torch.backends.cudnn.benchmark,
           'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
           'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
           'model_sha256': sha256_file(model_dir / 'your_model.pt'),
           'runner_sha256': sha256_file(model_dir / 'runner.py'),
           'recipe_sha256': sha256_file(model_dir / 'models_export.json'),
           'inputs': {}, 'gpu_mem_used_mib_before': nvsmi('memory.used'),
           'gpu_temp_c_before': nvsmi('temperature.gpu')}
    status, detail = 'FAIL', ''
    try:
        led.open_attempt(attempt, out_dir, model_sha256=rec['model_sha256'],
                         source_sha256=sha256_file(HERE / 'score01_bench.py'))
        t0 = time.perf_counter()
        runner = load_runner(model_dir)
        rec['load_s'] = round(time.perf_counter() - t0, 3)
        for name in [s.strip() for s in a.inputs.split(',') if s.strip()]:
            x = load_input(name)
            entry = {'input_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                     'shape': list(x.shape), 'cold_ms': None, 'warmup_ms': [], 'timed_ms': []}
            calls = [('cold', 1), ('warmup', a.warmup), ('timed', a.timed)]
            first = True
            for kind, n in calls:
                for i in range(n):
                    led.reserve('top', 'bench_%s_p%d_%s_%s_%d'
                                % (a.label, os.getpid(), name, kind, i + 1))
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    y = runner.infer(x, prompt='', timing={})
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t) * 1000.0
                    if not bool(torch.isfinite(y).all()):
                        raise RuntimeError('%s 输出含非有限值' % name)
                    if kind == 'cold':
                        entry['cold_ms'] = round(ms, 3)
                        entry['cold_shape'] = list(y.shape)
                        entry['cold_dtype'] = str(y.dtype)
                        entry['cold_finite'] = True
                        entry['cold_min'] = float(y.min())
                        entry['cold_max'] = float(y.max())
                    elif kind == 'warmup':
                        entry['warmup_ms'].append(round(ms, 3))
                    else:
                        entry['timed_ms'].append(round(ms, 3))
                    first = False
            tm = np.asarray(entry['timed_ms'], dtype=np.float64)
            entry['median_ms'] = round(float(np.median(tm)), 3)
            entry['mean_ms'] = round(float(tm.mean()), 3)
            entry['p90_ms'] = round(float(np.percentile(tm, 90)), 3)
            entry['min_ms'] = round(float(tm.min()), 3)
            entry['max_ms'] = round(float(tm.max()), 3)
            entry['std_ms'] = round(float(tm.std(ddof=1)), 3) if tm.size > 1 else None
            rec['inputs'][name] = entry
            print('[bench %s] %-8s cold=%.1f median=%.1f p90=%.1f mean=%.1f'
                  % (a.label, name, entry['cold_ms'], entry['median_ms'],
                     entry['p90_ms'], entry['mean_ms']), flush=True)
        rec['vram_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 1048576, 1)
        rec['vram_reserved_mib'] = round(torch.cuda.max_memory_reserved() / 1048576, 1)
        rec['gpu_mem_used_mib_after'] = nvsmi('memory.used')
        rec['gpu_temp_c_after'] = nvsmi('temperature.gpu')
        rec['wall_s'] = round(time.perf_counter() - t0, 3)
        rec['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        rec['status'] = 'OK'
        status = 'OK'
    except Exception as e:
        detail = '%s: %s' % (type(e).__name__, e)
        rec['status'] = 'FAIL'
        rec['error'] = detail
        print('[bench] FAIL %s' % detail, flush=True)
    finally:
        led.note('bench_%s_done' % a.label, {'status': rec.get('status')})
        led.close_attempt(status, note=detail)
        led.release_lock()
        (out_dir / ('bench_%s_p%d.json' % (a.label, os.getpid()))).write_text(
            json.dumps(rec, indent=1, ensure_ascii=False), encoding='utf-8')
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
