# -*- coding: utf-8 -*-
"""SCORE-01R 真实入口对照 bench（§0H 16/17/18/24/25/26）。

与 SCORE-01 版的区别（全部为测量修复，不改候选数值行为）：
  1. 真实无 profiling 入口：infer(x, prompt='')（不传 timing；spy 逐次记录实参为 None）；
  2. 两侧统一使用 A2 生成/验证的**冻结开关**（加载前设置并落盘）；
  3. 每次调用（含预热）**先扣后跑**；额度不足在加载模型之前即拒绝（0 次调用）；
  4. ABBA（第 1 轮）/ BAAB（第 2 轮）块顺序，每块每输入 2 预热 + 10 计时；
  5. 统计只用原始逐次数组；同时记录每次调用内部的 cuda 同步次数（v1 应为 6，v2 应为 0）。

用法: python score01r_bench.py <baseline_model_dir> <candidate_model_dir> <ledger> <out_dir>
                             --round 1|2 --attempt NAME
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os as _os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_mode  # noqa: E402
import r_ledger  # noqa: E402
import score01_bench as SB  # noqa: E402   （复用固定输入 BOX/load_input）
import score01r_protocol as P  # noqa: E402


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def nvsmi(field):
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=%s' % field,
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return None


def gpu_state():
    return {'temp_c': nvsmi('temperature.gpu'), 'sm_mhz': nvsmi('clocks.sm'),
            'power_w': nvsmi('power.draw'), 'mem_used_mib': nvsmi('memory.used'),
            'compute_apps': nvsmi('compute-apps')}


def side_id(model_dir):
    p = Path(model_dir)
    return {'model_dir': str(p), 'model_sha256': sha256_file(p / 'your_model.pt'),
            'runner_sha256': sha256_file(p / 'runner.py'),
            'recipe_sha256': sha256_file(p / 'models_export.json')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline_dir')
    ap.add_argument('candidate_dir')
    ap.add_argument('ledger')
    ap.add_argument('out_dir')
    ap.add_argument('--round', type=int, required=True, choices=(1, 2))
    ap.add_argument('--attempt', default=None)
    ap.add_argument('--inputs', default=','.join(P.INPUTS))
    a = ap.parse_args()

    inputs = [s.strip() for s in a.inputs.split(',') if s.strip()]
    order = P.block_order(a.round)
    need = len(order) * len(inputs) * (P.WARMUP + P.TIMED)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 冻结开关：必须在任何模型/CUDA 上下文之前设置 ──
    flags = frozen_mode.apply_frozen_mode()

    led = r_ledger.Ledger.from_file(a.ledger)
    attempt = a.attempt or ('score01r_round%d_%d' % (a.round, _os.getpid()))
    rec = {'schema': 'SCORE01R-bench/1', 'round': a.round, 'order': order,
           'warmup': P.WARMUP, 'timed': P.TIMED, 'inputs': inputs, 'attempt': attempt,
           'pid': _os.getpid(), 'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'flags': flags, 'frozen_mode': True, 'need_calls': need,
           'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu_name': torch.cuda.get_device_name(0),
           'bench_source_sha256': sha256_file(Path(__file__)),
           'protocol_source_sha256': sha256_file(HERE / 'score01r_protocol.py'),
           'side': {'A': side_id(a.baseline_dir), 'B': side_id(a.candidate_dir)},
           'gpu_before': gpu_state(), 'blocks': []}
    status, detail = 'FAIL', ''
    led.acquire_lock('SCORE-01R bench r%d' % a.round)
    try:
        # ── 额度预检（在加载模型之前；不足则 0 次调用）──
        d = led._read()
        rem = {k: int(led.limits[k]) - int(d['used'][k]) for k in ('top', 'top4k')}
        rec['budget_before'] = rem
        if rem['top'] < need:
            detail = 'BUDGET_INSUFFICIENT need=%d rem=%d' % (need, rem['top'])
            rec['status'] = 'BUDGET_INSUFFICIENT'
            print('[r%d] %s（未加载模型，0 次调用）' % (a.round, detail), flush=True)
            return 3
        led.open_attempt(attempt, out_dir,
                         source_sha256=rec['bench_source_sha256'],
                         model_sha256=rec['side']['B']['model_sha256'])

        # ── 计数包装（记录 runner 内部显式同步次数）──
        counter = {'n': 0}
        real_sync = torch.cuda.synchronize

        def counting_sync(*args, **kw):
            counter['n'] += 1
            return real_sync(*args, **kw)

        torch.cuda.synchronize = counting_sync
        seen_timing = []

        def load(tag, model_dir):
            t0 = time.perf_counter()
            r = SB.load_runner(model_dir)
            torch.cuda.synchronize()
            orig = r.infer

            def spy(x, prompt='', timing=None, **kw):
                seen_timing.append(timing)
                return orig(x, prompt=prompt, timing=timing, **kw)
            r.infer = spy
            rec.setdefault('load_s', {})[tag] = round(time.perf_counter() - t0, 3)
            return r

        runners = {}
        mode = 'interleaved'
        runners['A'] = load('A', a.baseline_dir)
        try:
            runners['B'] = load('B', a.candidate_dir)
        except RuntimeError as e:
            mode = 'serial_blocks'
            rec['interleaved_fallback'] = '%s: %s' % (type(e).__name__, e)
            torch.cuda.empty_cache()
        rec['mode'] = mode
        first_call = {}
        xs = {n: SB.load_input(n) for n in inputs}
        rec['input_sha256'] = {n: hashlib.sha256(xs[n].numpy().tobytes()).hexdigest()
                               for n in inputs}

        def run_block(label, block_idx):
            runner = runners.get(label)
            if runner is None:                     # 串行回落：本块单独加载
                runner = load(label, a.baseline_dir if label == 'A' else a.candidate_dir)
            tok = (a.round, _os.getpid(), label, block_idx)
            block = {'label': label, 'block_index': block_idx, 'gpu': gpu_state(),
                     'inputs': {}}
            for name in inputs:
                x = xs[name]
                entry = {'warmup_ms': [], 'timed_ms': [], 'sync_internal_warmup': [],
                         'sync_internal_timed': [], 'timing_arg_all_none': True}
                for phase, count in (('warmup', P.WARMUP), ('timed', P.TIMED)):
                    for i in range(count):
                        led.reserve('top', 'r%dp%d_%s_b%d_%s_%s_%d'
                                    % (tok[0], tok[1], label, block_idx, name, phase, i + 1))
                        counter['n'] = 0
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        y = runner.infer(x, prompt='')
                        torch.cuda.synchronize()
                        ms = (time.perf_counter() - t0) * 1000.0
                        if not bool(torch.isfinite(y).all()):
                            raise RuntimeError('输出含非有限值：%s/%s' % (label, name))
                        if seen_timing and seen_timing[-1] is not None:
                            entry['timing_arg_all_none'] = False
                        internal = counter['n'] - 2
                        if phase == 'warmup':
                            entry['warmup_ms'].append(round(ms, 3))
                            entry['sync_internal_warmup'].append(internal)
                        else:
                            entry['timed_ms'].append(round(ms, 3))
                            entry['sync_internal_timed'].append(internal)
                        if label not in first_call:
                            first_call[label] = round(ms, 3)
                entry['input_shape'] = list(x.shape)
                block['inputs'][name] = entry
                st = P.stats(entry['timed_ms'])
                print('[r%d %s b%d] %-8s median=%.1f p90=%.1f sync_int=%s'
                      % (a.round, label, block_idx, name, st['median'], st['p90'],
                         sorted(set(entry['sync_internal_timed']))), flush=True)
            if runners.get(label) is None:
                del runner
                torch.cuda.empty_cache()
            return block

        for k, label in enumerate(order):
            rec['blocks'].append(run_block(label, k))
        rec['first_call_ms'] = first_call
        rec['timing_arg_values'] = sorted({str(v) for v in seen_timing})
        torch.cuda.synchronize = real_sync
        rec['gpu_after'] = gpu_state()
        rec['vram_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 1048576, 1)
        rec['vram_reserved_mib'] = round(torch.cuda.max_memory_reserved() / 1048576, 1)
        rec['metrics'] = P.round_metrics(rec['blocks'], inputs)
        rec['status'] = 'OK'
        rec['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        status = 'OK'
        print('[r%d] verdict=%s improvement=%.4f drift=%s p90_ok=%s'
              % (a.round, rec['metrics']['verdict'], rec['metrics']['median_improvement'],
                 rec['metrics']['drift'], rec['metrics']['p90_ok']), flush=True)
    except Exception as e:
        detail = '%s: %s' % (type(e).__name__, e)
        rec['error'] = detail
        print('[r%d] FAIL %s' % (a.round, detail), flush=True)
    finally:
        try:
            used = led._read()['used']
            rec['budget_after'] = used
            rec['calls_used'] = (used['top'] - int(rec.get('budget_before', {}).get('top', 0))
                                 if rec.get('budget_before') else None)
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
        (out_dir / ('score01r_round%d_p%d.json' % (a.round, _os.getpid()))).write_text(
            json.dumps(rec, indent=1, ensure_ascii=False), encoding='utf-8')
    return 0 if status == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
