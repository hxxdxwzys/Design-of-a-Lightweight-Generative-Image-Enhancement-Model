# -*- coding: utf-8 -*-
"""SCORE-01R CPU 回归：真实入口（timing=None）+ 先扣后跑 + 冻结开关 + 协议判定。

覆盖 §0H 16/18/26 与新协议；不需要 GPU。
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
EB = HERE.parent
if str(EB) not in sys.path:
    sys.path.insert(0, str(EB))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V1 = _load('score01r_v1', EB / 'delivery_runner.py')
V2 = _load('score01r_v2', EB / 'score01_runner_v2.py')
P = _load('score01r_proto', EB / 'score01r_protocol.py')
BENCH = EB / 'score01r_bench.py'
PROFILE = EB / 'score01r_profile.py'


class Stub(torch.nn.Module):
    def __init__(self, tag=''):
        super().__init__()
        self.tag = tag
        self.calls = []

    def encode(self, x, tile_px):
        self.calls.append(('encode', tuple(x.shape)))
        n = int(tile_px) // 8
        return x.float().mean().expand(1, 4, n, n).clone()

    def features(self, x):
        self.calls.append(('features', tuple(x.shape)))
        return x.float().mean().expand(1, 1024, 768).clone()

    def step(self, lq, z, tc, tn, feats):
        self.calls.append(('step', tuple(lq.shape)))
        return lq * 0.5 + z * 0.25

    def decode(self, z, tile):
        self.calls.append(('decode', tuple(z.shape)))
        n = int(tile) * 8
        return z.float().mean().expand(1, 3, n, n).clone()

    def post(self, sr, src):
        self.calls.append(('post', tuple(sr.shape)))
        return (sr + src) * 0.5


def _model_dir(tmp_path, name='model_dir'):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'models_export.json').write_text(json.dumps({
        'weight_dtype': 'float16', 'model_file': 'your_model.pt', 'seed': 42,
        't_start': 1.0, 't_end': 0.0, 'dit_tile_size': 512, 'dit_tile_overlap_cli': 32,
        'vae_tile_size': 1024, 'vae_tile_overlap_cli': 32, 'patch_size': 2,
        'vae_enc_trace_px': [512, 1024], 'dec_trace_latent': [64, 128]}), encoding='utf-8')
    torch.jit.save(torch.jit.script(torch.nn.Module()), str(d / 'your_model.pt'))
    return d


# ── 1. 源码守卫：bench 不得再传 timing={} ──
def test_bench_source_uses_real_entry_without_timing_dict():
    src = BENCH.read_text(encoding='utf-8')
    assert 'timing={}' not in src, 'bench 不得传 timing={}（SC-R01 缺陷）'
    assert "runner.infer(x, prompt='')" in src
    assert 'frozen_mode.apply_frozen_mode()' in src
    psrc = PROFILE.read_text(encoding='utf-8')
    assert 'led.reserve' in psrc and psrc.index('led.reserve') < psrc.index('timing = {}')


# ── 2. 真实入口下 timing 实参为 None，且 v2 六处计时同步未触发 ──
@pytest.mark.parametrize('mod', [V2])
def test_default_entry_passes_none_and_skips_timing_syncs(tmp_path, mod):
    r = mod.Runner(str(_model_dir(tmp_path)), device='cpu')
    r.model = Stub()
    seen = {}
    orig = r._infer_body

    def spy(image_tensor, prompt='', timing=None):
        seen['timing'] = timing
        return orig(image_tensor, prompt=prompt, timing=timing)
    r._infer_body = spy
    c = {'n': 0}
    r._dev_sync = lambda c=c: c.__setitem__('n', c['n'] + 1)
    out = r.infer(torch.full((1, 3, 512, 512), 0.25), prompt='')
    assert seen['timing'] is None
    assert c['n'] == 0
    assert out.shape == (1, 3, 512, 512)


def test_profiling_entry_still_syncs_and_reports_stages(tmp_path):
    r = V2.Runner(str(_model_dir(tmp_path)), device='cpu')
    r.model = Stub()
    c = {'n': 0}
    r._dev_sync = lambda c=c: c.__setitem__('n', c['n'] + 1)
    timing = {}
    r.infer(torch.full((1, 3, 512, 512), 0.25), prompt='', timing=timing)
    assert c['n'] == 6
    assert set(timing) == {'encode_s', 'features_s', 'dit_s', 'decode_s',
                           'postprocess_s', 'infer_s'}


# ── 3. 先扣后跑：额度为零时 bench/profile 真实入口 0 次调用 ──
STUB_RUNNER = '''# -*- coding: utf-8 -*-
import os
from pathlib import Path


def _log(tag):
    p = os.environ.get('STUB_CALL_LOG')
    if p:
        with open(p, 'a', encoding='utf-8') as f:
            f.write(tag + '\\n')


class Runner:
    def __init__(self, model_dir):
        _log('load')

    def infer(self, x, prompt='', timing=None):
        _log('infer timing=%r' % (timing is None))
        import torch
        return torch.zeros_like(x)
'''


def _zero_budget_ledger(tmp_path):
    p = tmp_path / 'ledger_zero.json'
    p.write_text(json.dumps({
        'schema': 'EXPORT01R-ledger/1', 'created_at': '2026-09-18 00:00:00',
        'limits': {'top': 0, 'top4k': 0, 'component': 0, 'wall_s': 100000.0,
                   'disk_gib': 0.1},
        'prior_batch': {'old_ledgers': [], 'note': 'test', 'prior_top': 0,
                        'prior_top4k': 0, 'prior_component': 0},
        'used': {'top': 0, 'top4k': 0, 'component': 0}, 'events': [], 'attempts': [],
        'open_attempt': None}), encoding='utf-8')
    return p


def _stub_model_dir(tmp_path, name):
    d = _model_dir(tmp_path, name)
    (d / 'runner.py').write_text(STUB_RUNNER, encoding='utf-8')
    return d


@pytest.mark.parametrize('script,args,need', [
    (BENCH, lambda b, c, l, o: [str(b), str(c), str(l), str(o), '--round', '1',
                                '--attempt', 'cpu_zero'], 144),
    (PROFILE, lambda b, c, l, o: [str(b), str(l), str(o), '--tag', 'baseline'], 6)])
def test_zero_budget_makes_no_model_calls(tmp_path, script, args, need):
    b = _stub_model_dir(tmp_path, 'base')
    c = _stub_model_dir(tmp_path, 'cand')
    led = _zero_budget_ledger(tmp_path)
    out = tmp_path / 'out'
    log = tmp_path / 'stub_calls.txt'
    env = dict(os.environ)
    env['STUB_CALL_LOG'] = str(log)
    env['PYTHONIOENCODING'] = 'utf-8'
    cp = subprocess.run([sys.executable, '-B', str(script)] + args(b, c, led, out),
                        capture_output=True, text=True, timeout=600, env=env, cwd=str(EB),
                        encoding='utf-8', errors='replace')
    assert cp.returncode == 3, (cp.returncode, cp.stdout[-500:], cp.stderr[-500:])
    assert not log.exists() or log.read_text(encoding='utf-8').strip() == ''
    files = list(out.glob('*.json'))
    assert files, '应落盘 BUDGET_INSUFFICIENT 证据'
    rec = json.loads(files[0].read_text(encoding='utf-8'))
    assert rec['status'] == 'BUDGET_INSUFFICIENT'
    assert rec['need_calls'] == need
    assert not (tmp_path / 'ledger_zero.json.lock').exists(), '锁必须释放'


# ── 4. 协议：块顺序、统计、漂移与门槛 ──
def test_protocol_orders():
    assert P.block_order(1) == ['A', 'B', 'B', 'A']
    assert P.block_order(2) == ['B', 'A', 'A', 'B']
    with pytest.raises(ValueError):
        P.block_order(3)


def test_protocol_stats_match_numpy():
    xs = [100.0 + i for i in range(10)]
    s = P.stats(xs)
    assert s['median'] == float(np.median(xs))
    assert abs(s['p90'] - float(np.percentile(xs, 90))) < 1e-9
    assert s['n'] == 10


def _round(a_block_vals, b_block_vals, inputs=P.INPUTS, n=10):
    """构造一轮 block 记录：参数为长度 2 的列表（每块一个计时值，套用到三个输入）。"""
    labels = ['A', 'B', 'B', 'A']
    ai = bi = 0
    out = []
    for k, label in enumerate(labels):
        if label == 'A':
            v = a_block_vals[ai]
            ai += 1
        else:
            v = b_block_vals[bi]
            bi += 1
        out.append({'label': label, 'block_index': k,
                    'inputs': {nm: {'timed_ms': [float(v)] * n} for nm in inputs}})
    return out


def test_protocol_gate_pass_and_no_gain_and_all_blocks_used():
    # B 的两块一块极快一块极慢：汇总后中位数=150ms，改善为负 ⇒ 证明未挑最快块
    m = P.round_metrics(_round([100.0, 100.0], [100.0, 200.0]))
    assert m['verdict'] == 'NO_GAIN'
    assert m['median_improvement'] < -0.4, m['median_improvement']
    m2 = P.round_metrics(_round([100.0, 100.0], [90.0, 90.0]))
    assert m2['verdict'] == 'PASS' and m2['median_improvement'] >= 0.10


def test_protocol_drift_invalid_and_final_verdict():
    m = P.round_metrics(_round([100.0, 120.0], [80.0, 80.0]))
    assert m['verdict'] == 'INVALID' and m['drift'] > P.DRIFT_INVALID
    assert P.final_verdict([m, m])['verdict'] == 'INCONCLUSIVE'
    ok = P.round_metrics(_round([100.0, 100.0], [90.0, 90.0]))
    assert P.final_verdict([ok, ok])['verdict'] == 'VALID_PASS'
    ng = P.round_metrics(_round([100.0, 100.0], [99.0, 99.0]))
    assert P.final_verdict([ng, ng])['verdict'] == 'VALID_NO_GAIN'
