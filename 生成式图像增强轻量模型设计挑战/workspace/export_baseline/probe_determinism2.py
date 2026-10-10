# -*- coding: utf-8 -*-
"""组件级诊断（第二轮）：确定性开关与 autocast 对“运行间可复现性”和时延的影响。

对照配方（全部只用组件级小输入，不重跑 4K）：
  M1 fp16 权重/激活（基准，已知不确定）
  M2 fp16 + torch.use_deterministic_algorithms(True, warn_only=True)
  M3 fp16 + cudnn.deterministic=True, cudnn.benchmark=False
  M4 fp32 权重 + autocast(fp16)
每个配方对同一组件跑两次比较位相等（vae_enc_512 / dit_64 / dec_64），并记录 vae_enc_512 的时延中位数。

用法: python probe_determinism2.py <run_dir>
产出: <run_dir>/evidence/determinism_probe2.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline_baseline'))

import core as PBcore  # noqa: E402
from chain import Decode, DinoFeat, DitStep, VAEEncode, patch_fp16_embedders  # noqa: E402


def main():
    run = Path(sys.argv[1]).resolve()
    ev = run / 'evidence'
    dev = torch.device('cuda')
    cfg = PBcore.load_config()
    out = {'schema': 'EXPORT01-determinism-probe2/1', 'modes': {},
           'n_component_forwards': 0, 'wall_s': 0.0}
    t_start = time.perf_counter()
    assets = PBcore.Assets(cfg, dev)

    def build(dtype):
        merged = assets.model.merge_and_unload().to(dtype)
        if dtype == torch.float16:
            patch_fp16_embedders(merged, torch.float16)
        return (VAEEncode(assets.vae.to(dtype)).to(dev).eval(),
                Decode(assets.dec.to(dtype), assets.vae.config.scaling_factor).to(dev).eval(),
                DitStep(merged).to(dev).eval())

    def measure(enc, dit, dtype, tag, autocast=False, n=3):
        x = torch.randn(1, 3, 512, 512, device=dev, dtype=dtype) * 0.3
        z = torch.randn(1, 4, 64, 64, device=dev, dtype=dtype)
        tcur = torch.full((1,), 1.0, device=dev, dtype=dtype)
        tnxt = torch.full((1,), 0.0, device=dev, dtype=dtype)
        feats = torch.randn(1, 1024, 768, device=dev, dtype=dtype)
        rec = {}
        ctx = torch.autocast('cuda', dtype=torch.float16) if autocast else torch.no_grad()

        def two(fn):
            with torch.no_grad():
                a = fn()
                b = fn()
            d = (a.float() - b.float()).abs()
            return {'exact_equal': bool(torch.equal(a, b)), 'max_abs': float(d.max())}

        with ctx:
            try:
                rec['vae_enc_512'] = two(lambda: enc(x))
            except Exception as e:
                rec['vae_enc_512'] = {'error': '%s: %s' % (type(e).__name__, str(e)[:200])}
        try:
            rec['dit_64'] = two(lambda: dit(z, z, tcur, tnxt, feats))
        except Exception as e:
            rec['dit_64'] = {'error': '%s: %s' % (type(e).__name__, str(e)[:200])}
        # 时延（诊断，只测 vae_enc_512）
        try:
            torch.cuda.synchronize()
            ts = []
            for _ in range(n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with (torch.autocast('cuda', dtype=torch.float16) if autocast else torch.no_grad()):
                    enc(x)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            rec['vae_enc_512_median_s'] = round(statistics.median(ts), 4)
        except Exception as e:
            rec['vae_enc_512_median_s'] = None
            rec['timing_error'] = str(e)[:200]
        out['modes'][tag] = rec
        out['n_component_forwards'] += 6 + n

    # M1 基准
    enc, dec, dit = build(torch.float16)
    measure(enc, dit, torch.float16, 'M1_fp16')
    del enc, dec, dit
    torch.cuda.empty_cache()

    # M2 确定性算法
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        enc, dec, dit = build(torch.float16)
        measure(enc, dit, torch.float16, 'M2_fp16_deterministic_algorithms')
        del enc, dec, dit
        torch.cuda.empty_cache()
    except Exception as e:
        out['modes']['M2_fp16_deterministic_algorithms'] = {'error': str(e)[:300]}
    finally:
        torch.use_deterministic_algorithms(False)

    # M3 cudnn deterministic
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        enc, dec, dit = build(torch.float16)
        measure(enc, dit, torch.float16, 'M3_fp16_cudnn_deterministic')
        del enc, dec, dit
        torch.cuda.empty_cache()
    except Exception as e:
        out['modes']['M3_fp16_cudnn_deterministic'] = {'error': str(e)[:300]}
    finally:
        torch.backends.cudnn.deterministic = False

    # M4 fp32 + autocast
    enc, dec, dit = build(torch.float32)
    measure(enc, dit, torch.float32, 'M4_fp32_autocast', autocast=True)
    del enc, dec, dit
    torch.cuda.empty_cache()

    out['wall_s'] = round(time.perf_counter() - t_start, 3)
    out['peak_vram_mib'] = round(torch.cuda.max_memory_allocated() / 1048576, 1)
    (ev / 'determinism_probe2.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
