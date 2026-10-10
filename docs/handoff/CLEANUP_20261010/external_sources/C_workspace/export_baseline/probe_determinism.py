# -*- coding: utf-8 -*-
"""判定 FAIL 后的**组件级**诊断探针（不重跑 4K 顶层层级、不放宽任何判据）。

目的：把“纯 fp16 运行间不确定”定位到具体子模块，并测同一子模块在
      ①fp16 权重+fp16 激活  ②fp32 权重 + autocast(fp16)  两种配方下的可复现性。

用法: python probe_determinism.py <run_dir>
产出: <run_dir>/evidence/determinism_probe.json
预算口径：组件导出探针（≤64 次小前向），不消耗 4K 顶层层级额度。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline_baseline'))

import core as PBcore  # noqa: E402
from chain import Decode, DinoFeat, DitStep, VAEEncode, WaveletFix, patch_fp16_embedders  # noqa: E402


def pair_diff(fn, n=2):
    outs = []
    for _ in range(n):
        outs.append(fn())
    a, b = outs[0], outs[-1]
    if isinstance(a, (list, tuple)):
        a, b = a[0], b[0]
    d = (a.float() - b.float()).abs()
    return {'exact_equal': bool(torch.equal(a, b)), 'max_abs': float(d.max()),
            'mean_abs': float(d.mean()), 'shape': list(a.shape), 'dtype': str(a.dtype)}


def main():
    run = Path(sys.argv[1]).resolve()
    ev = run / 'evidence'
    dev = torch.device('cuda')
    cfg = PBcore.load_config()
    t0 = time.perf_counter()
    assets = PBcore.Assets(cfg, dev)
    out = {'schema': 'EXPORT01-determinism-probe/1',
           'note': '判定 FAIL 后的组件级诊断；不改变判据、不重跑 4K 顶层层级；输入尺寸逐项记录',
           'flags': {'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
                     'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
                     'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
                     'matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                     'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)},
           'passes': {}, 'n_component_forwards': 0, 'setup_s': round(time.perf_counter() - t0, 3)}

    def build(dtype):
        merged = assets.model.merge_and_unload()
        merged = merged.to(dtype)
        patched = patch_fp16_embedders(merged, torch.float16) if dtype == torch.float16 else []
        vae = assets.vae.to(dtype)
        dec = assets.dec.to(dtype)
        dino = assets.venc.to(dtype)
        return (VAEEncode(vae).to(dev).eval(), Decode(dec, vae.config.scaling_factor).to(dev).eval(),
                DinoFeat(dino, size=int(cfg['expected_dinov2_size']),
                         layer_index=int(cfg['expected_layer_dinov2b_list'][0]),
                         n_blocks=len(dino.blocks)).to(dev).eval(),
                DitStep(merged).to(dev).eval(), patched)

    def runs(enc, dnc, dino, dit, dtype, tag, include_1024=True):
        tcur = torch.full((1,), 1.0, device=dev, dtype=dtype)
        tnxt = torch.full((1,), 0.0, device=dev, dtype=dtype)
        x512 = torch.randn(1, 3, 512, 512, device=dev, dtype=dtype) * 0.3
        x1024 = torch.randn(1, 3, 1024, 1024, device=dev, dtype=dtype) * 0.3 if include_1024 else None
        z = torch.randn(1, 4, 64, 64, device=dev, dtype=dtype)
        z128 = torch.randn(1, 4, 128, 128, device=dev, dtype=dtype)
        feats = torch.randn(1, 1024, 768, device=dev, dtype=dtype)
        wav = WaveletFix(sigma=float(cfg['align_sigma'])).to(dev).eval()
        res = {}
        with torch.no_grad():
            res['vae_enc_512'] = pair_diff(lambda: enc(x512))
            if x1024 is not None:
                res['vae_enc_1024'] = pair_diff(lambda: enc(x1024))
            res['dino_448_from_512'] = pair_diff(lambda: dino(x512))
            res['dit_64'] = pair_diff(lambda: dit(z, z, tcur, tnxt, feats))
            res['dec_64'] = pair_diff(lambda: dnc(z))
            res['dec_128'] = pair_diff(lambda: dnc(z128))
            res['wavelet_512'] = pair_diff(lambda: wav(x512, x512))
        n = sum(2 for _ in res)
        out['passes'][tag] = res
        out['n_component_forwards'] += n
        del wav, x512, x1024, z, z128, feats
        torch.cuda.empty_cache()

    enc, dnc, dino, dit, patched = build(torch.float16)
    out['passes']['patched_embedders'] = patched
    runs(enc, dnc, dino, dit, torch.float16, 'fp16_weights_fp16_activations')
    del enc, dnc, dino, dit
    torch.cuda.empty_cache()

    enc, dnc, dino, dit, _ = build(torch.float32)
    with torch.autocast('cuda', dtype=torch.float16):
        runs(enc, dnc, dino, dit, torch.float32, 'fp32_weights_autocast_fp16',
             include_1024=False)
    del enc, dnc, dino, dit
    torch.cuda.empty_cache()
    out['wall_s'] = round(time.perf_counter() - t0, 3)
    out['peak_vram_mib'] = round(torch.cuda.max_memory_allocated() / 1048576, 1)
    (ev / 'determinism_probe.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                               encoding='utf-8')
    print(json.dumps({'passes': {k: {kk: vv['max_abs'] for kk, vv in v.items()}
                                 for k, v in out['passes'].items() if isinstance(v, dict)},
                      'n_forwards': out['n_component_forwards'], 'wall_s': out['wall_s'],
                      'flags': out['flags']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
