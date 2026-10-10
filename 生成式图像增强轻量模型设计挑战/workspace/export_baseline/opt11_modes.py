# -*- coding: utf-8 -*-
"""OPT-11 执行侧候选扫描：channels_last / optimize_for_inference / cudnn.benchmark。
用法: python opt11_modes.py frozen|native
每个 arm 都记录首调、稳态中位数（8 次）与相对基线的 uint8 最大差/差异像素比。
"""
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

MODE = sys.argv[1] if len(sys.argv) > 1 else 'frozen'
MODEL_DIR = r'D:\models\opt10\model_dir'
RUN = Path(r'D:\models\opt11')
RUN.mkdir(parents=True, exist_ok=True)
LED = RUN / 'ledger_opt11.json'
X = SB.load_input('c1_512')
TAG = time.strftime('%H%M%S')

flags = frozen_mode.apply_frozen_mode() if MODE == 'frozen' else frozen_mode.read_flags()
led = r_ledger.Ledger.from_file(str(LED)) if LED.is_file() else None
if led is None:
    raise SystemExit('ledger 缺失')

def to_u8(t):
    return torch.round((t.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu()

def build(arm):
    R = SB.load_runner(MODEL_DIR)
    t0 = time.perf_counter()
    if arm in ('cl', 'cl_bench'):
        try:
            R.model = R.model.to(memory_format=torch.channels_last)
        except Exception:
            R.model = R.model.apply(lambda m: m.to(memory_format=torch.channels_last))
    elif arm == 'opt':
        R.model = torch.jit.optimize_for_inference(R.model)
    if arm.endswith('bench'):
        torch.backends.cudnn.benchmark = True
    return R, time.perf_counter() - t0

def run(R, x):
    tm = {}
    t = time.perf_counter()
    out = R.infer(x, prompt='', timing=tm)
    return out, tm, (time.perf_counter() - t) * 1000.0

def prep_x(x, arm):
    return x if not arm.startswith('cl') else x.contiguous(memory_format=torch.channels_last)

stale = (led._read() or {}).get('open_attempt')
if stale:
    led.close_attempt('ABORTED', '上轮异常退出的残留 attempt（脚本自愈）')
    print('[warn] 已关闭残留 attempt:', stale.get('attempt_id'), flush=True)

arms = ['base', 'cl', 'opt'] if MODE == 'frozen' else ['base', 'base_bench', 'cl']
res = {'mode': MODE, 'flags': flags, 'arms': {}}
ref_u8 = None
print('== mode=%s ==' % MODE, flush=True)
print('%-11s %8s %9s %9s %9s %9s %9s %9s %8s' %
      ('arm', 'load_s', 'first_ms', 'enc', 'feat', 'dit', 'dec', 'median', 'u8max'), flush=True)
for arm in arms:
    if arm.endswith('bench'):
        torch.backends.cudnn.benchmark = True
    R, load_s = build('base' if arm == 'base_bench' else arm)
    if arm == 'base_bench':
        torch.backends.cudnn.benchmark = True
    x = prep_x(X, arm)
    led.acquire_lock('OPT-11 modes %s %s' % (MODE, arm))
    led.open_attempt('opt11_%s_%s_%d' % (MODE, arm, os.getpid()), RUN,
                     source_sha256=SB.sha256_file(Path(__file__)))
    try:
        for i in range(2):
            led.reserve('top', 'opt11_%s_%s_%s_w%d' % (MODE, arm, TAG, i))
            out, tm, w = run(R, x)
        first = w
        rows, outs = [], []
        for i in range(8):
            led.reserve('top', 'opt11_%s_%s_%s_t%d' % (MODE, arm, TAG, i))
            out, tm, w = run(R, x)
            rows.append({'enc': tm['encode_s'] * 1e3, 'feat': tm['features_s'] * 1e3,
                         'dit': tm['dit_s'] * 1e3, 'dec': tm['decode_s'] * 1e3, 'wall': w})
            outs.append(to_u8(out))
        med = lambda k: statistics.median([r[k] for r in rows])  # noqa: E731
        m = {k: med(k) for k in ('enc', 'feat', 'dit', 'dec', 'wall')}
        if ref_u8 is None:
            ref_u8 = outs[-1]
            u8max, frac = 0, 0.0
        else:
            d = (outs[-1].to(torch.int16) - ref_u8.to(torch.int16)).abs()
            u8max = int(d.max())
            frac = float((d.max(dim=1).values > 0).float().mean())
        res['arms'][arm] = {'load_s': load_s, 'first_ms': first, 'median': m,
                            'u8max': u8max, 'diff_frac': frac,
                            'out_sha': __import__('hashlib').sha256(outs[-1].numpy().tobytes()).hexdigest()[:16]}
        print('%-11s %8.2f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %8d  diff=%.3f' %
              (arm, load_s, first, m['enc'], m['feat'], m['dit'], m['dec'], m['wall'], u8max, frac), flush=True)
        led.close_attempt('DONE', 'modes arm %s/%s' % (MODE, arm))
    finally:
        led.release_lock()
    torch.backends.cudnn.benchmark = False
p = RUN / ('modes_%s.json' % MODE)
p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
print('[done] ->', p, flush=True)
