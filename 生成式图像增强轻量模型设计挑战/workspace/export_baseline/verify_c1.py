# -*- coding: utf-8 -*-
"""C1 重载验证：fp16 源参考（eager）与重载 ScriptModule 分两个独立实例先后执行。

协议（与 R-B 一致的“先后加载、释放前者再加载后者”）：
  阶段一 源参考实例：4 个固定输入各跑一次，输出落盘 .npy（GPU 内存释放前的唯一大对象）
  阶段二 重载实例：同一批输入各跑一次，与阶段一逐元素比较；512 裁切额外交错重复一次
  阶段三 只读对照：与已验收 fp32（R-B attempt_01）NPY 的差异（诊断项，不作为导出判据）

用法: python verify_c1.py <run_dir>
产出: <run_dir>/evidence/{paired_readings_c1.json, c1_verdict.json, src_outputs/*.npy}
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline_baseline'))

import core as PBcore  # noqa: E402
import delivery_runner as DR  # noqa: E402
from chain import Decode, DinoFeat, DitStep, VAEEncode, WaveletFix, patch_fp16_embedders  # noqa: E402

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


class EagerBundle(torch.nn.Module):
    """源参考：同权重同链但不 trace，用于隔离“导出/重载”这一变量。"""

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
    assets = PBcore.Assets(cfg, dev)
    merged = assets.model.merge_and_unload().half()
    patched = patch_fp16_embedders(merged, torch.float16)
    bundle = EagerBundle(VAEEncode(assets.vae.half()).to(dev).eval(),
                         Decode(assets.dec.half(), assets.vae.config.scaling_factor).to(dev).eval(),
                         DinoFeat(assets.venc.half(), size=int(cfg['expected_dinov2_size']),
                                  layer_index=int(cfg['expected_layer_dinov2b_list'][0]),
                                  n_blocks=len(assets.venc.blocks)).to(dev).eval(),
                         DitStep(merged).to(dev).eval(),
                         WaveletFix(sigma=float(cfg['align_sigma'])).to(dev).eval())
    return bundle, patched


def to_u8(t):
    return torch.clamp(t.detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5, 0, 1).mul(255).to(torch.uint8)


def compare_case(name, src_npy, out, fp32_ref_path=None):
    """成对比较（所有设备搬运集中在此，CPU/GPU 输入都可）。

    out 与 src 必须形状一致；fp32 参考若形状不符则跳过并记 skipped_reason。
    """
    src = torch.from_numpy(np.load(src_npy, allow_pickle=False)).to(out.device)
    if tuple(src.shape) != tuple(out.shape):
        raise ValueError('源参考与重载输出形状不一致: %s vs %s'
                         % (tuple(src.shape), tuple(out.shape)))
    d = (src.float() - out.float()).abs()
    rec = {'name': name,
           'same_precision_max_abs': float(d.max()),
           'same_precision_mean_abs': float(d.mean()),
           'png_pixel_identical': bool(torch.equal(to_u8(src), to_u8(out))),
           'out_shape': list(out.shape), 'out_dtype': str(out.dtype),
           'out_finite': bool(torch.isfinite(out).all()),
           'out_min': float(out.min()), 'out_max': float(out.max()),
           'out_sha256': hashlib.sha256(
               out.detach().cpu().float().numpy().tobytes()).hexdigest()}
    if fp32_ref_path is not None and Path(fp32_ref_path).is_file():
        ref = torch.from_numpy(np.load(fp32_ref_path, allow_pickle=False))
        if tuple(ref.shape) == tuple(out.shape):
            refd = ref.to(out.device)
            dd = (refd.float() - out.float()).abs()
            rec['fp32_ref'] = {'path': str(fp32_ref_path), 'max_abs': float(dd.max()),
                               'mean_abs': float(dd.mean()),
                               'png_pixel_identical': bool(torch.equal(to_u8(ref), to_u8(out)))}
        else:
            rec['fp32_ref'] = {'path': str(fp32_ref_path), 'skipped': 'shape_mismatch',
                               'ref_shape': list(ref.shape)}
    return rec


def main():
    run = Path(sys.argv[1]).resolve()
    ev = run / 'evidence'
    srcdir = ev / 'src_outputs'
    srcdir.mkdir(parents=True, exist_ok=True)
    prereg_p = run / 'prereg_c1.json'
    prereg = json.loads(prereg_p.read_text(encoding='utf-8'))
    prereg_sha = sha256_file(prereg_p)
    exp_cfg = json.loads((HERE / 'export_config.json').read_text(encoding='utf-8'))
    tol = exp_cfg['tolerances']
    dev = torch.device('cuda')
    model_dir = run / 'model_dir'
    readings = {
        'schema': 'EXPORT01-c1-readings/1', 'prereg_sha256': prereg_sha,
        'model_sha256': sha256_file(model_dir / 'your_model.pt'),
        'phase1': {}, 'cases': [],
    }

    def load_input(case):
        im = Image.open(case['source_file']).convert('RGB')
        if case['crop_box']:
            im = im.crop(tuple(case['crop_box']))
        arr = np.asarray(im)
        x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return x * 2.0 - 1.0

    inputs = {c['name']: load_input(c) for c in prereg['cases']}

    # ── 阶段一：源参考实例（或复用同一预登记下已落盘的 phase-1 产物）──
    reuse = '--reuse-src' in sys.argv
    if reuse:
        prov = json.loads((srcdir / 'phase1_provenance.json').read_text(encoding='utf-8'))
        bad = [n for n, m in prov['files'].items()
               if not (srcdir / n).is_file() or sha256_file(srcdir / n) != m['sha256']]
        if bad:
            raise SystemExit('REUSE_HASH_MISMATCH: %s' % bad)
        if prov.get('runner_recipe_sha256') and \
                prov['runner_recipe_sha256'] != sha256_file(model_dir / 'models_export.json'):
            raise SystemExit('REUSE_RECIPE_MISMATCH: phase-1 产物与当前 recipe 不同源')
        readings['phase1'] = {'source': 'REUSED_FROM_ABORTED_ATTEMPT_PHASE1',
                              'provenance': prov, 'freed_before_phase2': True}
        print('[phase1] reused 4 source outputs (hash-verified)', flush=True)
    else:
        t0 = time.perf_counter()
        cfg = PBcore.load_config()
        bundle, patched = build_eager(cfg, dev)
        eager_runner = EagerRunner(str(model_dir), bundle, device='cuda')
        readings['phase1']['patched_embedders'] = patched
        readings['phase1']['setup_s'] = round(time.perf_counter() - t0, 3)
        for case in prereg['cases']:
            x = inputs[case['name']]
            timing = {}
            t = time.perf_counter()
            out = eager_runner.infer(x, timing=timing)
            rec = {'wall_s': round(time.perf_counter() - t, 3), 'timing': timing}
            np.save(srcdir / ('%s_src.npy' % case['name']),
                    out.detach().cpu().float().numpy())
            rec['npy'] = '%s_src.npy' % case['name']
            rec['npy_sha256'] = sha256_file(srcdir / rec['npy'])
            readings['phase1'][case['name']] = rec
            print('[phase1] %s  %.2fs' % (case['name'], rec['wall_s']), flush=True)
        del eager_runner, bundle
        gc.collect()
        torch.cuda.empty_cache()
        readings['phase1']['freed_before_phase2'] = True
        readings['phase1']['vram_after_free_mib'] = round(
            torch.cuda.memory_allocated() / 1048576, 1)

    # ── 阶段二：重载实例 ──
    t0 = time.perf_counter()
    re_runner = DR.Runner(str(model_dir), device='cuda')
    readings['phase2_setup_s'] = round(time.perf_counter() - t0, 3)
    def flush():
        (ev / 'paired_readings_c1.json').write_text(
            json.dumps(readings, indent=1, ensure_ascii=False), encoding='utf-8')

    for case in prereg['cases']:
        x = inputs[case['name']]
        timing = {}
        try:
            t = time.perf_counter()
            out = re_runner.infer(x, timing=timing)
            wall = round(time.perf_counter() - t, 3)
            rec = compare_case(case['name'], srcdir / ('%s_src.npy' % case['name']), out,
                               FP32_REF.get(case['name']))
            rec.update({'wh': case['wh'], 'wall_s': wall, 'timing': timing,
                        'input_sha256': hashlib.sha256(
                            x.detach().cpu().numpy().tobytes()).hexdigest()})
            if case['name'] == 'case1_crop512':
                t = time.perf_counter()
                rep = re_runner.infer(x)
                rec['repeat_wall_s'] = round(time.perf_counter() - t, 3)
                rec['repeat_max_abs'] = float((rep.float() - out.float()).abs().max())
                rec['repeat_png_identical'] = bool(torch.equal(to_u8(rep), to_u8(out)))
        except Exception as e:                       # 单例失败不丢失已完成的读数
            rec = {'name': case['name'], 'wh': case['wh'], 'error': '%s: %s'
                   % (type(e).__name__, e)}
            print('[phase2] %s  ERROR %s' % (case['name'], rec['error']), flush=True)
        readings['cases'].append(rec)
        flush()
        if 'error' not in rec:
            print('[phase2] %s  %.2fs  max=%.3e mean=%.3e png_eq=%s'
                  % (case['name'], rec['wall_s'], rec['same_precision_max_abs'],
                     rec['same_precision_mean_abs'], rec['png_pixel_identical']), flush=True)

    ok = all(c['same_precision_max_abs'] <= tol['same_precision_max_abs'] and
             c['same_precision_mean_abs'] <= tol['same_precision_mean_abs'] and
             c['png_pixel_identical'] and c['out_finite'] and c['out_dtype'] == 'torch.float32'
             for c in readings['cases'])
    diag_ok = all((c.get('fp32_ref') is None) or
                  (c['fp32_ref']['max_abs'] <= tol['fp16_vs_fp32_max_abs_diagnostic'] and
                   c['fp32_ref']['mean_abs'] <= tol['fp16_vs_fp32_mean_abs_diagnostic'])
                  for c in readings['cases'])
    readings['totals'] = {
        'top_level_infers': len(readings['cases']) * 2 + 1,
        'n_4k_infers': 4,
        'peak_vram_mib': round(torch.cuda.max_memory_allocated() / 1048576, 1),
    }
    verdict = {'schema': 'EXPORT01-c1-verdict/1',
               'same_precision_pass': bool(ok),
               'fp16_vs_fp32_diagnostic_pass': bool(diag_ok),
               'verdict': 'PASS' if (ok and diag_ok) else 'FAIL',
               'tolerances': tol, 'model_sha256': readings['model_sha256'],
               'prereg_sha256': prereg_sha, 'totals': readings['totals']}
    (ev / 'paired_readings_c1.json').write_text(json.dumps(readings, indent=1, ensure_ascii=False),
                                                encoding='utf-8')
    (ev / 'c1_verdict.json').write_text(json.dumps(verdict, indent=1, ensure_ascii=False),
                                        encoding='utf-8')
    print(json.dumps(verdict, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
