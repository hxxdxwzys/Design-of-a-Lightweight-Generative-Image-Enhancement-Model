# -*- coding: utf-8 -*-
"""OPT-11 剖面：定位 512 测速路径分段成本（墙钟分段 + CUDA event 的 GPU 忙时间）。"""
import os
import sys
import json
import time
import statistics
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import frozen_mode  # noqa: E402
import r_ledger  # noqa: E402
import score01_bench as SB  # noqa: E402

MODEL_DIR = r'D:\models\opt10\model_dir'
RUN = Path(r'D:\models\opt11')
RUN.mkdir(parents=True, exist_ok=True)
LED = RUN / 'ledger_opt11.json'
INPUTS = ('c1_512', 'c7_512', 'c25_512')

flags = frozen_mode.apply_frozen_mode()
led = r_ledger.Ledger(str(LED), top=400, top4k=40, component=0, wall_s=7200.0, disk_gib=8.0,
                      prior={'old_ledgers': [{'ledger': r'D:\models\opt10\ledger_opt10.json',
                                              'used': {'top': 125, 'top4k': 105, 'component': 0},
                                              'note': '只读引用'}],
                             'note': 'OPT-11 剖面/提速：top<=400, 4K<=40, component=0, 监督<=2h, 磁盘<=8GiB',
                             'prior_top': 125, 'prior_top4k': 105, 'prior_component': 0})
led.acquire_lock('OPT-11 profile')
try:
    led.open_attempt('opt11_profile_%d' % os.getpid(), RUN,
                     source_sha256=SB.sha256_file(Path(__file__)))
    t0 = time.perf_counter()
    R = SB.load_runner(MODEL_DIR)
    load_s = time.perf_counter() - t0
    print('[load] jit.load+构造 = %.2f s' % load_s, flush=True)

    xs = {n: SB.load_input(n) for n in INPUTS}
    rec = {'schema': 'OPT11-profile/1', 'model_dir': MODEL_DIR, 'flags': flags,
           'load_s': load_s, 'started_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'inputs': {}}
    print('%-8s %8s %8s %8s %8s %8s %8s %8s' %
          ('input', 'enc', 'feat', 'dit', 'dec', 'post', 'wall', 'gpu_busy'), flush=True)
    for n, x in xs.items():
        first = None
        for i in range(2):
            led.reserve('top', 'opt11_prof_%s_warm%d' % (n, i))
            tm = {}
            t = time.perf_counter()
            R.infer(x, prompt='', timing=tm)
            w = (time.perf_counter() - t) * 1000.0
            if first is None:
                first = (tm.get('infer_s', 0) * 1000.0, w)
        rows = []
        for i in range(10):
            led.reserve('top', 'opt11_prof_%s_t%d' % (n, i))
            tm = {}
            t = time.perf_counter()
            R.infer(x, prompt='', timing=tm)
            w = (time.perf_counter() - t) * 1000.0
            rows.append({'enc': tm['encode_s'] * 1000, 'feat': tm['features_s'] * 1000,
                         'dit': tm['dit_s'] * 1000, 'dec': tm['decode_s'] * 1000,
                         'post': tm['postprocess_s'] * 1000, 'infer': tm['infer_s'] * 1000,
                         'wall': w})
        busy = []
        for i in range(3):
            led.reserve('top', 'opt11_prof_%s_ev%d' % (n, i))
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            s.record(); R.infer(x, prompt='', ); e.record(); torch.cuda.synchronize()
            busy.append(s.elapsed_time(e))
        med = lambda k: statistics.median([r[k] for r in rows])  # noqa: E731
        rec['inputs'][n] = {'first_call_ms': first, 'rows': rows,
                            'median_ms': {k: med(k) for k in ('enc', 'feat', 'dit', 'dec', 'post', 'infer', 'wall')},
                            'gpu_busy_ms': busy, 'gpu_busy_median': statistics.median(busy)}
        m = rec['inputs'][n]['median_ms']
        print('%-8s %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f' %
              (n, m['enc'], m['feat'], m['dit'], m['dec'], m['post'], m['wall'],
               statistics.median(busy)), flush=True)
        print('    first_call: %.1f ms (infer_s) / %.1f ms (wall)' % first, flush=True)
    tot = statistics.mean([rec['inputs'][n]['median_ms']['infer'] for n in INPUTS])
    busy_tot = statistics.mean([rec['inputs'][n]['gpu_busy_median'] for n in INPUTS])
    rec['mean_infer_ms'] = tot
    rec['mean_gpu_busy_ms'] = busy_tot
    rec['cpu_overhead_ms'] = tot - busy_tot
    print('[summary] 三输入均值 infer=%.1f ms, GPU busy=%.1f ms, 差(CPU/launch+IO)=%.1f ms'
          % (tot, busy_tot, tot - busy_tot), flush=True)
    (RUN / 'profile_opt10.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding='utf-8')
    print('[done] ->', RUN / 'profile_opt10.json', flush=True)
    led.close_attempt('DONE', 'profile ok')
finally:
    led.release_lock()
