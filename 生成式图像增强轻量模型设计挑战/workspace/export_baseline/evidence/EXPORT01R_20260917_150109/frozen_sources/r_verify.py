# -*- coding: utf-8 -*-
"""EXPORT-01R 复验序列（在 worker 子进程内执行；本模块不做进程/锁管理）。

固定顺序（§0A，12 次顶层层级上限）：
  A(独立 eager 实例) : case1 512 连续3次 + case1 511x513 1次
  B(独立重载 JIT 实例): 同样 4 次，512 每次即时对照 A 与自身重复，通过才推进
  4K: case1 原图 A/B 各1次 → 即时对照；通过后 case7 原图 A/B 各1次
每次调用前 Ledger.reserve('top', ...) 先扣后跑；首个失败立即返回并落盘。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_PB = Path(__file__).resolve().parent.parent / 'pipeline_baseline'
if str(_PB) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(_PB))

import delivery_runner as DR  # noqa: E402
from chain import (Decode, DinoFeat, DitStep, VAEEncode, WaveletFix,  # noqa: E402
                   patch_fp16_embedders)

Image.MAX_IMAGE_PIXELS = None
RB_NPY = Path('D:/models/pipeline_baseline/PIPE01RB_20260917_021500/attempt_01')
FP32_REF = {
    'case1_crop512': RB_NPY / 'attempt_01_072821_A_case1_crop512_r1.npy',
    'case1_full': RB_NPY / 'attempt_01_072821_A_case1_full_r2.npy',
    'case7_full': RB_NPY / 'attempt_01_072821_A_case7_full_r3.npy',
}


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def to_u8(t):
    tt = torch.clamp(t.detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5, 0, 1)
    return tt.mul(255).to(torch.uint8)


class Recorder(torch.nn.Module):
    """只观测不改变行为：记录各阶段输出的哈希（首个调用额外保留原始张量）。"""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.stages = {}
        self.keep_raw = False

    def _rec(self, name, t):
        self.stages.setdefault(name, {'sha256': tensor_sha(t), 'shape': list(t.shape),
                                      'dtype': str(t.dtype),
                                      'finite': bool(torch.isfinite(t).all())})
        if self.keep_raw and name not in self.raw:
            self.raw[name] = t.detach().float().cpu().clone()

    def reset(self, keep_raw=False):
        self.stages = {}
        self.raw = {}
        self.keep_raw = keep_raw

    def encode(self, x, tile_px):
        t = self.inner.encode(x, tile_px)
        self._rec('encode', t)
        return t

    def features(self, x):
        t = self.inner.features(x)
        self._rec('features', t)
        return t

    def step(self, lq, z, tc, tn, feats):
        self._rec('noise', z)
        t = self.inner.step(lq, z, tc, tn, feats)
        self._rec('step', t)
        return t

    def decode(self, z, tile):
        t = self.inner.decode(z, tile)
        self._rec('decode_raw', t)
        return t

    def post(self, sr, src):
        t = self.inner.post(sr, src)
        self._rec('post', t)
        return t


class EagerBundle(torch.nn.Module):
    def __init__(self, enc, dec, dino, dit, wav):
        super().__init__()
        self.enc, self.dec, self.dino, self.dit, self.wav = enc, dec, dino, dit, wav

    def encode(self, x, tile_px):
        return self.enc(x)

    def features(self, x):
        return self.dino(x)

    def step(self, lq, z, tc, tn, feats):
        return self.dit(lq, z, tc, tn, feats)

    def decode(self, z, tile_latent):
        return self.dec(z)

    def post(self, sr, src):
        return self.wav(sr, src)


class EagerRunner(DR.Runner):
    def __init__(self, model_dir, bundle, device='cuda'):
        super().__init__(model_dir, device=device)
        self.model = bundle


def build_eager(cfg, dev):
    import core as PBcore
    assets = PBcore.Assets(cfg, dev)
    merged = assets.model.merge_and_unload().to(torch.float16)
    patched = patch_fp16_embedders(merged, torch.float16)
    bundle = EagerBundle(VAEEncode(assets.vae.to(torch.float16)).to(dev).eval(),
                         Decode(assets.dec.to(torch.float16),
                                assets.vae.config.scaling_factor).to(dev).eval(),
                         DinoFeat(assets.venc.to(torch.float16),
                                  size=int(cfg['expected_dinov2_size']),
                                  layer_index=int(cfg['expected_layer_dinov2b_list'][0]),
                                  n_blocks=len(assets.venc.blocks)).to(dev).eval(),
                         DitStep(merged).to(dev).eval(),
                         WaveletFix(sigma=float(cfg['align_sigma'])).to(dev).eval())
    return bundle, patched


def load_input(case):
    im = Image.open(case['source_file']).convert('RGB')
    if case.get('crop_box'):
        im = im.crop(tuple(case['crop_box']))
    arr = np.asarray(im)
    x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x * 2.0 - 1.0


def compare(out, ref_npy, tag):
    ref = torch.from_numpy(np.load(ref_npy, allow_pickle=False)).to(out.device)
    if tuple(ref.shape) != tuple(out.shape):
        raise RuntimeError('%s 形状不一致 %s vs %s' % (tag, tuple(ref.shape), tuple(out.shape)))
    d = (ref.float() - out.float()).abs()
    return {'max_abs': float(d.max()), 'mean_abs': float(d.mean()),
            'png_pixel_identical': bool(torch.equal(to_u8(ref), to_u8(out)))}


def check(crit, tol, label):
    ok = (crit['max_abs'] <= tol['same_precision_max_abs'] and
          crit['mean_abs'] <= tol['same_precision_mean_abs'] and crit['png_pixel_identical'])
    if not ok:
        raise RuntimeError('%s 未达同精度判据: max=%.3e mean=%.3e png_eq=%s'
                           % (label, crit['max_abs'], crit['mean_abs'],
                              crit['png_pixel_identical']))
    return True


def run_verification(run_dir, ledger, prereg_path, model_dir, tol, exp_dir):
    """返回 readings；失败抛异常（调用方落盘 FAIL 并关闭 attempt）。"""
    run_dir = Path(run_dir)
    model_dir = Path(model_dir)
    exp_dir = Path(exp_dir)
    # 大产物（双方与重复 NPY）只放 C 盘运行目录；仓库 evidence 只留小证据
    npy_dir = Path(run_dir) / 'npy'
    npy_dir.mkdir(parents=True, exist_ok=True)
    readings_npy_root = npy_dir
    prereg = json.loads(Path(prereg_path).read_text(encoding='utf-8'))
    cases = {c['name']: c for c in prereg['cases']}
    dev = torch.device('cuda')
    readings = {'schema': 'EXPORT01R-readings/1',
                'prereg_sha256': sha256_file(prereg_path),
                'model_sha256': sha256_file(model_dir / 'your_model.pt'),
                'order': [], 'phaseA': {}, 'phaseB': {}, 'pairs': [], 'migration': [],
                'execution_mode': {'cublas_workspace_config': __import__('os').environ.get(
                    'CUBLAS_WORKSPACE_CONFIG'),
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'cudnn_deterministic': torch.backends.cudnn.deterministic,
                    'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                    'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                    'jit_optimized_execution': False}}

    # ── EXP-R05 合并对照：两个独立进程、同精度同输入、只改"是否合并"（先扣后跑）──
    import subprocess
    import sys as _sys
    ledger.reserve('component', 'merge_probe_peft_instance')
    ledger.reserve('component', 'merge_probe_merged_instance')
    mp = subprocess.run([_sys.executable, '-B',
                         str(Path(__file__).resolve().parent / 'r_merge_probe.py'),
                         'drive', str(exp_dir)], capture_output=True, text=True, timeout=1800)
    mpp = exp_dir / 'merge_probe.json'
    readings['merge_probe'] = (json.loads(mpp.read_text(encoding='utf-8'))
                               if mpp.is_file() else
                               {'error': (mp.stderr or '')[-600:], 'exit': mp.returncode})
    if 'error' in readings['merge_probe']:
        raise RuntimeError('合并对照探针失败：%s' % readings['merge_probe']['error'])

    def save_npy(tag, t):
        p = npy_dir / ('%s.npy' % tag)
        np.save(p, t.detach().cpu().float().numpy())
        return {'npy': str(p), 'npy_sha256': sha256_file(p),
                'tensor_sha256': tensor_sha(t), 'shape': list(t.shape)}

    # ── 阶段 A：独立 eager 实例 ──
    import core as PBcore
    cfg = PBcore.load_config()
    bundle, patched = build_eager(cfg, dev)
    a_runner = EagerRunner(str(model_dir), bundle, device='cuda')
    rec_a = Recorder(a_runner.model).eval()
    a_runner.model = rec_a
    readings['phaseA']['patched_embedders'] = patched
    for i in range(1, 4):
        case = cases['case1_crop512']
        x = load_input(case)
        ledger.note('A_512_%d_prepare' % i, {'input_sha256': tensor_sha(x)})
        ledger.reserve('top', 'A_512_%d' % i)
        rec_a.reset(keep_raw=(i == 1))
        with torch.jit.optimized_execution(False):
            t0 = time.perf_counter()
            out = a_runner.infer(x, timing={})
            wall = round(time.perf_counter() - t0, 3)
        rec = {'call': 'A_512_%d' % i, 'wall_s': wall, 'stages': rec_a.stages,
               'input_sha256': tensor_sha(x), **save_npy('A_512_r%d' % i, out)}
        if i == 1:
            for k, v in rec_a.raw.items():
                np.save(npy_dir / ('A_512_r1_stage_%s.npy' % k), v.numpy())
            rec['stage_raw'] = sorted(rec_a.raw)
        readings['phaseA']['call_%d' % i] = rec
        readings['order'].append(rec['call'])
    case = cases['case1_crop511x513']
    x = load_input(case)
    ledger.reserve('top', 'A_511x513')
    rec_a.reset()
    with torch.jit.optimized_execution(False):
        t0 = time.perf_counter()
        out = a_runner.infer(x, timing={})
        wall = round(time.perf_counter() - t0, 3)
    readings['phaseA']['call_4'] = {'call': 'A_511x513', 'wall_s': wall, 'stages': rec_a.stages,
                                    'input_sha256': tensor_sha(x),
                                    **save_npy('A_511x513', out)}
    readings['order'].append('A_511x513')
    del a_runner, bundle, rec_a
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    readings['phaseA']['released'] = True

    # ── 阶段 B：独立重载 JIT 实例 ──
    b_runner = DR.Runner(str(model_dir), device='cuda')
    rec_b = Recorder(b_runner.model).eval()
    b_runner.model = rec_b
    b_512 = []
    for i in range(1, 4):
        x = load_input(cases['case1_crop512'])
        ledger.reserve('top', 'B_512_%d' % i)
        rec_b.reset(keep_raw=(i == 1))
        with torch.jit.optimized_execution(False):
            t0 = time.perf_counter()
            out = b_runner.infer(x, timing={})
            wall = round(time.perf_counter() - t0, 3)
        crit_a = compare(out, npy_dir / ('A_512_r%d.npy' % i), 'B_512_%d vs A' % i)
        crit_self = {'max_abs': 0.0, 'mean_abs': 0.0, 'png_pixel_identical': True}
        if b_512:
            crit_self = compare(out, npy_dir / 'B_512_r1.npy', 'B_512_%d vs B_512_r1' % i)
        rec = {'call': 'B_512_%d' % i, 'wall_s': wall, 'stages': rec_b.stages,
               'input_sha256': tensor_sha(x), 'vs_A': crit_a, 'vs_first_B512': crit_self,
               **save_npy('B_512_r%d' % i, out)}
        if i == 1:
            for k, v in rec_b.raw.items():
                np.save(npy_dir / ('B_512_r1_stage_%s.npy' % k), v.numpy())
            rec['stage_raw'] = sorted(rec_b.raw)
        readings['phaseB']['call_%d' % i] = rec
        readings['order'].append(rec['call'])
        check(crit_a, tol, 'B_512_%d vs A' % i)
        if i > 1:
            check(crit_self, tol, 'B_512_%d vs B_512_r1' % i)
        b_512.append(out.detach().cpu())
    x = load_input(cases['case1_crop511x513'])
    ledger.reserve('top', 'B_511x513')
    rec_b.reset()
    with torch.jit.optimized_execution(False):
        t0 = time.perf_counter()
        out = b_runner.infer(x, timing={})
        wall = round(time.perf_counter() - t0, 3)
    crit = compare(out, npy_dir / 'A_511x513.npy', 'B_511x513 vs A')
    readings['phaseB']['call_4'] = {'call': 'B_511x513', 'wall_s': wall, 'stages': rec_b.stages,
                                    'input_sha256': tensor_sha(x), 'vs_A': crit,
                                    **save_npy('B_511x513', out)}
    readings['order'].append('B_511x513')
    check(crit, tol, 'B_511x513 vs A')

    # ── 4K：case1 与 case7，各实例顺序加载/释放 ──
    for name in ('case1_full', 'case7_full'):
        # A：重建 eager 实例（A 已释放）
        bundle, _ = build_eager(cfg, dev)
        a_runner = EagerRunner(str(model_dir), bundle, device='cuda')
        x = load_input(cases[name])
        ledger.reserve('top', '%s_A' % name, is_4k=True)
        with torch.jit.optimized_execution(False):
            t0 = time.perf_counter()
            out_a = a_runner.infer(x, timing={})
            wall_a = round(time.perf_counter() - t0, 3)
        rec_a = {'call': '%s_A' % name, 'wall_s': wall_a, 'input_sha256': tensor_sha(x),
                 **save_npy('%s_A' % name, out_a)}
        del a_runner, bundle
        gc.collect()
        torch.cuda.empty_cache()
        # B：重载实例
        br = DR.Runner(str(model_dir), device='cuda')
        ledger.reserve('top', '%s_B' % name, is_4k=True)
        with torch.jit.optimized_execution(False):
            t0 = time.perf_counter()
            out_b = br.infer(x, timing={})
            wall_b = round(time.perf_counter() - t0, 3)
        rec_b2 = {'call': '%s_B' % name, 'wall_s': wall_b, 'input_sha256': tensor_sha(x),
                  **save_npy('%s_B' % name, out_b)}
        del br
        gc.collect()
        torch.cuda.empty_cache()
        crit = compare(out_b, npy_dir / ('%s_A.npy' % name), '%s B vs A' % name)
        readings['pairs'].append({'name': name, 'A': rec_a, 'B': rec_b2, 'pair': crit})
        readings['order'].extend([rec_a['call'], rec_b2['call']])
        check(crit, tol, '%s B vs A' % name)
        ref = FP32_REF.get(name)
        if ref is not None and ref.is_file():
            mig = compare(out_b, ref, '%s fp16 vs fp32' % name)
            readings['migration'].append({'name': name, 'fp32_ref': str(ref), **mig})

    ref = FP32_REF.get('case1_crop512')
    if ref is not None and ref.is_file():
        out = torch.from_numpy(np.load(npy_dir / 'B_512_r1.npy', allow_pickle=False))
        mig = compare(out, ref, 'case1_crop512 fp16 vs fp32')
        readings['migration'].append({'name': 'case1_crop512', 'fp32_ref': str(ref), **mig})
    return readings
