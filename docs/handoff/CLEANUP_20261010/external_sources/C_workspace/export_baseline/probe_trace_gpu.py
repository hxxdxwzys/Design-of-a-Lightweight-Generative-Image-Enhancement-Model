# -*- coding: utf-8 -*-
"""C1 探针：GPU 上构建 fp16 数值链、trace 子模块、容器 script、save+reload 并做首次等价性对照。

用法: python probe_trace_gpu.py <out_dir>
产出: <out_dir>/probe_result.json + your_model.pt（探针产物，非最终交付）
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PB = HERE.parent / 'pipeline_baseline'
sys.path.insert(0, str(PB))
sys.path.insert(0, str(HERE))

import core as PBcore  # noqa: E402
from chain import (Decode, DinoFeat, DitStep, ExportBundle, VAEEncode,  # noqa: E402
                   WaveletFix, patch_fp16_embedders)


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


_LOG = None


def stage(msg):
    line = '[stage] %s  %.1fs' % (msg, time.perf_counter() - _T0)
    print(line, flush=True)
    if _LOG is not None:
        _LOG.write(line + '\n')
        _LOG.flush()


_T0 = time.perf_counter()


def main():
    global _LOG
    import faulthandler
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _LOG = open(out / 'probe_stages.log', 'w', encoding='utf-8', buffering=1)
    faulthandler.dump_traceback_later(120, repeat=True, file=_LOG)
    stage('start')
    log = {'out_dir': str(out)}
    dev = torch.device('cuda')
    t0 = time.perf_counter()
    cfg = PBcore.load_config()
    assets = PBcore.Assets(cfg, dev)
    log['assets_loaded_s'] = round(time.perf_counter() - t0, 3)
    stage('assets_loaded')
    log['env'] = PBcore.env_fingerprint()

    import numpy as np
    from PIL import Image

    # ── 合并 LoRA（fp32 下合并，再整体转 fp16）──
    t = time.perf_counter()
    model = assets.model
    merged = model.merge_and_unload()
    log['lora_merge_s'] = round(time.perf_counter() - t, 3)
    log['merged_type'] = type(merged).__name__
    log['has_forward_flexible'] = bool(hasattr(merged, 'forward_flexible'))

    # 合并等价性（fp32、小输入、组件探针）
    dev_small = torch.device('cuda')
    x_small = torch.randn(1, 8, 64, 64, device=dev_small, dtype=torch.float32) * 0.1
    tt = torch.full((1,), 1.0, device=dev_small)
    rr = torch.full((1,), 0.0, device=dev_small)
    feat_small = torch.randn(1, 1024, 768, device=dev_small, dtype=torch.float32) * 0.1
    with torch.no_grad():
        u_peft = model(torch.cat([x_small[:, :4], x_small[:, :4]], 1), tt, rr, [feat_small])
        u_merged = merged.forward_flexible(x_small, tt, rr, [feat_small])
    log['lora_merge_max_abs_diff'] = float((u_peft - u_merged).abs().max())
    log['lora_merge_mean_abs_diff'] = float((u_peft - u_merged).abs().mean())
    del u_peft, u_merged, model, x_small, feat_small
    stage('lora_merge_checked')
    torch.cuda.empty_cache()

    # ── fp16 转换 ──
    vae = assets.vae.half()
    dec = assets.dec.half()
    dino = assets.venc.half()
    merged = merged.half()
    log['patched_embedders'] = patch_fp16_embedders(merged, torch.float16)
    stage('fp16_cast_done')
    n_blocks = len(dino.blocks)
    wrap_enc = VAEEncode(vae).to(dev).eval()
    wrap_dec = Decode(dec, vae.config.scaling_factor).to(dev).eval()
    wrap_dino = DinoFeat(dino, size=int(cfg['expected_dinov2_size']),
                         layer_index=int(cfg['expected_layer_dinov2b_list'][0]),
                         n_blocks=n_blocks).to(dev).eval()
    wrap_dit = DitStep(merged).to(dev).eval()
    wrap_wav = WaveletFix(sigma=float(cfg['align_sigma'])).to(dev).eval()
    log['dino_blocks'] = n_blocks

    def rnd(*shape):
        return torch.randn(*shape, device=dev, dtype=torch.float16)

    # ── trace ──
    traces = {}
    t = time.perf_counter()
    torch.cuda.empty_cache()
    traces['vae_enc_512'] = torch.jit.trace(wrap_enc, rnd(1, 3, 512, 512), strict=False,
                                           check_trace=False)
    stage('traced vae_enc_512')
    torch.cuda.empty_cache()
    traces['vae_enc_1024'] = torch.jit.trace(wrap_enc, rnd(1, 3, 1024, 1024), strict=False,
                                            check_trace=False)
    stage('traced vae_enc_1024')
    torch.cuda.empty_cache()
    traces['dino_feat'] = torch.jit.trace(wrap_dino, rnd(1, 3, 512, 512), strict=False,
                                         check_trace=False)
    stage('traced dino_feat')
    tcur = torch.full((1,), 1.0, device=dev, dtype=torch.float16)
    tnxt = torch.full((1,), 0.0, device=dev, dtype=torch.float16)
    traces['dit_step'] = torch.jit.trace(
        wrap_dit, (rnd(1, 4, 64, 64), rnd(1, 4, 64, 64), tcur, tnxt, rnd(1, 1024, 768)),
        strict=False, check_trace=False)
    stage('traced dit_step')
    torch.cuda.empty_cache()
    traces['dec_64'] = torch.jit.trace(wrap_dec, rnd(1, 4, 64, 64), strict=False,
                                      check_trace=False)
    stage('traced dec_64')
    traces['dec_128'] = torch.jit.trace(wrap_dec, rnd(1, 4, 128, 128), strict=False,
                                       check_trace=False)
    stage('traced dec_128')
    traces['wavelet'] = torch.jit.trace(wrap_wav, (rnd(1, 3, 64, 64), rnd(1, 3, 64, 64)),
                                        strict=False, check_trace=False)
    stage('traced wavelet')
    log['trace_s'] = round(time.perf_counter() - t, 3)
    log['traced'] = sorted(traces)

    # ── 容器 script + save + reload ──
    t = time.perf_counter()
    bundle = ExportBundle(traces['vae_enc_512'], traces['vae_enc_1024'], traces['dino_feat'],
                          traces['dit_step'], traces['dec_64'], traces['dec_128'],
                          traces['wavelet'])
    scripted = torch.jit.script(bundle)
    stage('scripted container')
    ckpt = out / 'probe_model.pt'
    torch.jit.save(scripted, str(ckpt))
    log['script_save_s'] = round(time.perf_counter() - t, 3)
    log['ckpt_bytes'] = ckpt.stat().st_size
    log['ckpt_sha256'] = sha256_file(ckpt)
    stage('saved + hashed ckpt')
    reloaded = torch.jit.load(str(ckpt), map_location=dev).eval()
    log['reload_ok'] = True
    stage('reloaded')

    # ── 成对等价性：fp16 eager 源参考 vs 重载 bundle（case1 中心 512 裁切）──
    test_dir = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集')
    im = Image.open(test_dir / 'case1.jpg').convert('RGB')
    W, H = im.size
    box = [(W - 512) // 2, (H - 512) // 2, (W - 512) // 2 + 512, (H - 512) // 2 + 512]
    crop = im.crop(box)
    arr = np.asarray(crop).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = (x * 2.0 - 1.0).to(dev, dtype=torch.float16)
    log['crop_box'] = box
    log['input_sha256'] = hashlib.sha256(np.asarray(crop).tobytes()).hexdigest()

    def run_eager(x_in):
        g = torch.Generator(device=dev)
        g.manual_seed(42)
        with torch.no_grad():
            lq = wrap_enc(x_in)
            feats = wrap_dino(x_in)
            z = torch.randn(lq.shape, generator=g, device=dev, dtype=torch.float32).half()
            z1 = wrap_dit(lq, z, tcur, tnxt, feats)
            sr = wrap_dec(z1)
            return wrap_wav(sr, x_in), sr

    def run_reloaded(x_in):
        g = torch.Generator(device=dev)
        g.manual_seed(42)
        with torch.no_grad():
            lq = reloaded.encode(x_in, 512)
            feats = reloaded.features(x_in)
            z = torch.randn(lq.shape, generator=g, device=dev, dtype=torch.float32).half()
            z1 = reloaded.step(lq, z, tcur, tnxt, feats)
            sr = reloaded.decode(z1, 64)
            return reloaded.post(sr, x_in), sr

    t = time.perf_counter()
    out_e, sr_e = run_eager(x)
    log['eager_512_s'] = round(time.perf_counter() - t, 3)
    stage('eager 512 done')
    t = time.perf_counter()
    out_r, sr_r = run_reloaded(x)
    log['reloaded_512_s'] = round(time.perf_counter() - t, 3)
    stage('reloaded 512 done')
    log['sr_dtype'] = str(sr_r.dtype)
    log['sr_max_abs_diff'] = float((sr_e.float() - sr_r.float()).abs().max())
    d = (out_e.float() - out_r.float()).abs()
    log['out_max_abs_diff'] = float(d.max())
    log['out_mean_abs_diff'] = float(d.mean())
    log['out_shape'] = list(out_r.shape)
    log['out_finite'] = bool(torch.isfinite(out_r).all())
    log['out_min'] = float(out_r.min())
    log['out_max'] = float(out_r.max())
    log['peak_vram_mib'] = round(torch.cuda.max_memory_allocated() / 1048576, 1)

    (out / 'probe_result.json').write_text(json.dumps(log, indent=1, ensure_ascii=False),
                                           encoding='utf-8')
    print(json.dumps(log, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
