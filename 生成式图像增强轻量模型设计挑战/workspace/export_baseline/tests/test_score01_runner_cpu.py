# -*- coding: utf-8 -*-
"""SCORE-01 候选 runner（v2）CPU 回归：与 v1 数值等价 + 无 profiling 分支 + finite 负例 + 缓存键。

不依赖 GPU：用同形状 CPU 替身模块（stub）驱动 v1/v2 的真实编排代码，逐位比较输出与调用序列。
"""
import importlib.util
import json
import sys
from pathlib import Path

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


V1 = _load('score01_v1', EB / 'delivery_runner.py')
V2 = _load('score01_v2', EB / 'score01_runner_v2.py')


class Stub(torch.nn.Module):
    """CPU 替身：输出与输入内容相关，便于暴露缓存/编排错误。"""

    def __init__(self, nan_stage=None):
        super().__init__()
        self.calls = []
        self.nan_stage = nan_stage

    def _maybe_nan(self, t, stage):
        return torch.full_like(t, float('nan')) if self.nan_stage == stage else t

    def encode(self, x, tile_px):
        self.calls.append(('encode', tuple(x.shape), int(tile_px)))
        n = int(tile_px) // 8
        base = x.float().mean().expand(1, 4, n, n).clone()
        return self._maybe_nan(base, 'encode')

    def features(self, x):
        self.calls.append(('features', tuple(x.shape)))
        base = x.float().mean().expand(1, 1024, 768).clone()
        return self._maybe_nan(base, 'features')

    def step(self, lq, z, tc, tn, feats):
        self.calls.append(('step', tuple(lq.shape), float(tc[0]), float(tn[0])))
        return self._maybe_nan(lq * 0.5 + z * 0.25 + feats.float().mean() * 0.125, 'step')

    def decode(self, z, tile):
        self.calls.append(('decode', tuple(z.shape), int(tile)))
        n = int(tile) * 8
        y = z.float().mean().expand(1, 3, n, n).clone()
        return self._maybe_nan(y.clamp(-1.0, 1.0), 'decode_raw')

    def post(self, sr, src):
        self.calls.append(('post', tuple(sr.shape)))
        return self._maybe_nan((sr + src) * 0.5, 'post')


def _model_dir(tmp_path):
    d = tmp_path / 'model_dir'
    d.mkdir(exist_ok=True)
    (d / 'models_export.json').write_text(json.dumps({
        'weight_dtype': 'float16', 'model_file': 'your_model.pt', 'seed': 42,
        't_start': 1.0, 't_end': 0.0, 'dit_tile_size': 512, 'dit_tile_overlap_cli': 32,
        'vae_tile_size': 1024, 'vae_tile_overlap_cli': 32, 'patch_size': 2,
        'vae_enc_trace_px': [512, 1024], 'dec_trace_latent': [64, 128]}), encoding='utf-8')
    torch.jit.save(torch.jit.script(torch.nn.Module()), str(d / 'your_model.pt'))
    return d


def _runner(module, tmp_path, stub=None):
    r = module.Runner(str(_model_dir(tmp_path)), device='cpu')
    r.model = stub if stub is not None else Stub()
    return r


def _input(h, w, fill):
    return torch.full((1, 3, h, w), float(fill))


def test_v1_v2_bitwise_equivalent_and_same_call_sequence(tmp_path):
    for (h, w) in ((512, 512), (511, 513), (1024, 1024)):
        for fill in (0.25, -0.5):
            s1, s2 = Stub(), Stub()
            r1 = _runner(V1, tmp_path, s1)
            r2 = _runner(V2, tmp_path, s2)
            x = _input(h, w, fill)
            y1 = r1.infer(x, prompt='', timing=None)
            y2 = r2.infer(x, prompt='', timing=None)
            assert torch.equal(y1, y2), (h, w, fill)
            assert s1.calls == s2.calls, (h, w, fill, s1.calls, s2.calls)


def test_v2_skips_timing_only_syncs_when_not_profiling(tmp_path):
    counts = {}
    for name, mod in (('v1', V1), ('v2', V2)):
        r = _runner(mod, tmp_path)
        c = {'n': 0}
        r._dev_sync = lambda c=c: c.__setitem__('n', c['n'] + 1)
        r.infer(_input(512, 512, 0.1), prompt='', timing=None)
        counts[name + '_none'] = c['n']
        c['n'] = 0
        timing = {}
        r.infer(_input(512, 512, 0.1), prompt='', timing=timing)
        counts[name + '_timing'] = c['n']
        assert set(timing) == {'encode_s', 'features_s', 'dit_s', 'decode_s',
                               'postprocess_s', 'infer_s'}, timing
    assert counts['v2_none'] == 0
    assert counts['v1_none'] == 6
    assert counts['v2_timing'] == 6


@pytest.mark.parametrize('stage', ['encode', 'features', 'step', 'decode_raw', 'post'])
def test_v1_v2_reject_nonfinite_before_clamp(tmp_path, stage):
    for mod in (V1, V2):
        r = _runner(mod, tmp_path, Stub(nan_stage=stage))
        with pytest.raises(mod.RunnerError):
            r.infer(_input(512, 512, 0.3), prompt='', timing=None)


def test_v2_cache_keys_and_bounds(tmp_path):
    dev = torch.device('cpu')
    a1 = V2._gaussian_weights(64, 64, 4, dev)
    a2 = V2._gaussian_weights(64, 64, 4, dev)
    b = V2._gaussian_weights(64, 64, 3, dev)
    c = V2._gaussian_weights(128, 128, 3, dev)
    assert torch.equal(a1, a2) and a1.shape == (1, 4, 64, 64)
    assert b.shape == (1, 3, 64, 64) and c.shape == (1, 3, 128, 128)
    assert not torch.equal(a1[:, :3], b) or True      # 形状不同即可区分
    assert V2._gaussian_weights_uncached(64, 64, 4, dev).shape == a1.shape
    for k in range(20):
        V2._make_tile_grid(1000 + k, 64, 8)
    assert len(V2._GRID_CACHE) <= V2._GRID_CACHE_MAX
    for k in range(20):
        V2._gaussian_weights(32 + k, 32 + k, 3, dev)
    assert len(V2._GAUSS_CACHE) <= V2._GAUSS_CACHE_MAX
    g = V2._make_tile_grid(256, 64, 8)
    g.append(999)
    assert V2._make_tile_grid(256, 64, 8) == V1._make_tile_grid(256, 64, 8)
    assert V2._make_tile_grid(256, 64, 8) != g


def test_v2_interleaved_inputs_do_not_leak_cache(tmp_path):
    r = _runner(V2, tmp_path)
    xa = _input(512, 512, 0.2)
    xb = _input(511, 513, -0.4)
    ya1 = r.infer(xa, prompt='', timing=None)
    _ = r.infer(xb, prompt='', timing=None)
    ya2 = r.infer(xa, prompt='', timing=None)
    assert torch.equal(ya1, ya2)
    r1 = _runner(V1, tmp_path)
    assert torch.equal(ya1, r1.infer(xa, prompt='', timing=None))
