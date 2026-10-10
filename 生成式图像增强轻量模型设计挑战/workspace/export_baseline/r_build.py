# -*- coding: utf-8 -*-
"""EXPORT-01R 导出构建（在 worker 子进程内执行，组件前向先扣后跑）。

与 EXPORT-01 的 build_export.py 区别（EXP-R06/R07 修复）：
  - 每次 trace/真实前向都经 Ledger.reserve('component', ...) 先扣后跑；
  - 转换记录写 **CastIn**（t_embedder.mlp / r_embedder.mlp），不再写 CastOut；
  - source_hashes 记录 VAE **实际加载权重与 config 的内容哈希**（不止文件计数）；
  - 产出 bundle_id 与 bundle_manifest（契约字段：parents/conversion/files/numeric_recipe）；
  - 结构改动、trace 特化尺寸、执行模式一并写进 numeric_recipe。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import sys

import torch

_PB = Path(__file__).resolve().parent.parent / 'pipeline_baseline'
if str(_PB) not in sys.path:
    sys.path.insert(0, str(_PB))

import core as PBcore  # noqa: E402
from chain import (Decode, DinoFeat, DitStep, ExportBundle, VAEEncode, WaveletFix,  # noqa: F401
                   patch_device_safe_timestep, patch_fp16_embedders)


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def tensor_content_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().to('cpu').float()
        h.update(k.encode('utf-8'))
        h.update(str(tuple(t.shape)).encode('utf-8'))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def bundle_id_of(parents, numeric_recipe):
    payload = json.dumps({'parents': sorted(
        [{'role': p['role'], 'sha256': p.get('sha256')} for p in parents],
        key=lambda x: x['role']), 'numeric_recipe': numeric_recipe}, sort_keys=True,
        ensure_ascii=False)
    return 'EXPORT01R-' + sha256_bytes(payload.encode('utf-8'))[:32]


def build(model_dir, run_dir, ledger, exp_dir, export_config, here, vae_fp32=False,
          full_fp32=False):
    model_dir = Path(model_dir)
    exp_dir = Path(exp_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg = PBcore.load_config()
    dev = torch.device('cuda')
    out = {'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}

    t0 = time.perf_counter()
    ledger.reserve('component', 'build_load_assets')
    assets = PBcore.Assets(cfg, dev)
    out['assets_load_s'] = round(time.perf_counter() - t0, 3)

    # ── 父资产与内容哈希（含 VAE 实际加载权重/config）──
    parents = []
    for role, path in (('base_weights', cfg['base_weights']), ('lora', cfg['lora_path']),
                       ('light_decoder', cfg['light_decoder']), ('dinov2_ckpt', cfg['dinov2_ckpt'])):
        p = Path(path)
        parents.append({'role': role, 'path': str(p), 'bytes': p.stat().st_size,
                        'sha256': sha256_file(p)})
    vae_content = {'vae_state_dict_sha256': tensor_content_sha(assets.vae.state_dict()),
                   'vae_config_sha256': sha256_bytes(json.dumps(
                       dict(assets.vae.config), sort_keys=True, default=str).encode('utf-8')),
                   'vae_dir': str(Path(cfg['ae_dir']) / 'vae'),
                   'n_state_keys': len(assets.vae.state_dict())}
    parents.append({'role': 'vae', 'path': str(Path(cfg['ae_dir']) / 'vae'),
                    **vae_content})
    base_content = {'base_state_dict_sha256': tensor_content_sha(assets.model.state_dict())}
    parents.append({'role': 'base_dit(peft, fp32)', 'path': cfg['base_weights'], **base_content})

    # ── 精度迁移：合并 LoRA → 时间嵌入 CastIn → fp16 ──
    work_dtype = torch.float32 if full_fp32 else torch.float16
    merged = assets.model.merge_and_unload().to(work_dtype)
    patched = patch_fp16_embedders(merged, work_dtype)
    # EXP-R13：先把时间嵌入的设备路径修好，再 trace（否则保存/重载后 CPU arange 与异设备常量混用）
    device_patched = patch_device_safe_timestep(merged)
    # DELIV-01R A2：编码器 / LightDecoder 可从父权重以 FP32 重新加载（不得由 fp16 升位冒充）
    vae = assets.vae.to(torch.float32) if (vae_fp32 or full_fp32) else assets.vae.to(torch.float16)
    dec = assets.dec.to(torch.float32) if (vae_fp32 or full_fp32) else assets.dec.to(torch.float16)
    vae_compute = torch.float32 if (vae_fp32 or full_fp32) else torch.float16
    dino = assets.venc.to(work_dtype)
    out['patched_embedders'] = patched
    out['device_safe_timestep'] = device_patched
    out['vae_fp32'] = bool(vae_fp32 or full_fp32)
    out['full_fp32'] = bool(full_fp32)
    _wd = 'fp32' if work_dtype == torch.float32 else 'fp16'
    _vd = 'fp32' if vae_compute == torch.float32 else 'fp16'
    out['module_dtypes'] = {'vae_encoder': str(vae_compute), 'light_decoder': str(vae_compute),
                            'dit': str(work_dtype), 'dino': str(work_dtype),
                            'boundary_casts': ['enc_in %s->%s' % (_wd, _vd),
                                               'latent %s->%s (进入 DiT 前)' % (_vd, _wd),
                                               'decode_in %s->%s' % (_wd, _vd),
                                               'decode_out ->fp32 (wavelet 保持 fp32)'],
                            'embedder_cast': 'CastIn(%s)（全 FP32 下为恒等变换，见 chain.CastIn）'
                                             % _wd,
                            'vae_posterior': 'mode（抽样关闭）'}

    wrap_enc = VAEEncode(vae, vae_compute, work_dtype).to(dev).eval()
    wrap_dec = Decode(dec, vae.config.scaling_factor, vae_compute, torch.float32).to(dev).eval()
    wrap_dino = DinoFeat(dino, size=int(cfg['expected_dinov2_size']),
                         layer_index=int(cfg['expected_layer_dinov2b_list'][0]),
                         n_blocks=len(dino.blocks)).to(dev).eval()
    wrap_dit = DitStep(merged).to(dev).eval()
    wrap_wav = WaveletFix(sigma=float(cfg['align_sigma'])).to(dev).eval()

    def rnd(*shape):
        return torch.randn(*shape, device=dev, dtype=work_dtype)

    tcur = torch.full((1,), float(cfg['t_start']), device=dev, dtype=work_dtype)
    tnxt = torch.full((1,), float(cfg['t_end']), device=dev, dtype=work_dtype)
    plan = [
        ('vae_enc_512', lambda: torch.jit.trace(wrap_enc, rnd(1, 3, 512, 512), strict=False,
                                                check_trace=False)),
        ('vae_enc_1024', lambda: torch.jit.trace(wrap_enc, rnd(1, 3, 1024, 1024), strict=False,
                                                 check_trace=False)),
        ('dino_feat', lambda: torch.jit.trace(wrap_dino, rnd(1, 3, 512, 512), strict=False,
                                              check_trace=False)),
        ('dit_step', lambda: torch.jit.trace(
            wrap_dit, (rnd(1, 4, 64, 64), rnd(1, 4, 64, 64), tcur, tnxt, rnd(1, 1024, 768)),
            strict=False, check_trace=False)),
        ('dec_64', lambda: torch.jit.trace(wrap_dec, rnd(1, 4, 64, 64), strict=False,
                                           check_trace=False)),
        ('dec_128', lambda: torch.jit.trace(wrap_dec, rnd(1, 4, 128, 128), strict=False,
                                            check_trace=False)),
        ('wavelet', lambda: torch.jit.trace(wrap_wav, (rnd(1, 3, 64, 64), rnd(1, 3, 64, 64)),
                                            strict=False, check_trace=False)),
    ]
    traces = {}
    t = time.perf_counter()
    for name, fn in plan:
        torch.cuda.empty_cache()
        ledger.reserve('component', 'trace_%s' % name)
        traces[name] = fn()
    out['trace_s'] = round(time.perf_counter() - t, 3)
    out['trace_component_forwards'] = len(plan)

    bundle = ExportBundle(traces['vae_enc_512'], traces['vae_enc_1024'], traces['dino_feat'],
                          traces['dit_step'], traces['dec_64'], traces['dec_128'],
                          traces['wavelet'])
    scripted = torch.jit.script(bundle)
    ckpt = model_dir / 'your_model.pt'
    torch.jit.save(scripted, str(ckpt))
    out['model_bytes'] = ckpt.stat().st_size
    out['model_sha256'] = sha256_file(ckpt)
    torch.jit.load(str(ckpt), map_location='cpu')
    out['reload_ok_cpu'] = True

    shutil.copyfile(Path(here) / 'delivery_runner.py', model_dir / 'runner.py')
    numeric_recipe = {
        'weight_dtype': ('float32' if work_dtype == torch.float32 else 'float16'),
        'postprocess_dtype': 'float32',
        'blend_accumulator_dtype': 'float32', 'seed': int(cfg['seed']),
        't_start': float(cfg['t_start']), 't_end': float(cfg['t_end']),
        'dit_tile_size': int(cfg['dit_tile_size']),
        'dit_tile_overlap_cli': int(cfg['dit_tile_overlap_cli']),
        'vae_tile_size': int(cfg['vae_tile_size']),
        'vae_tile_overlap_cli': int(cfg['vae_tile_overlap_cli']),
        'patch_size': int(cfg['expected_patch_size']),
        'align_sigma': int(cfg['align_sigma']), 'posterior_mode': True,
        'vae_enc_trace_px': [512, 1024], 'dec_trace_latent': [64, 128],
        'dinov2_input_size': int(cfg['expected_dinov2_size']),
        'noise_protocol': 'runner 局部 torch.Generator(seed=42) 生成 fp32 噪声后 .half() 显式传入',
        'fusion_semantics': 'DitStep 返回更新后的 z1；分块对 z1 做高斯加权融合（不再二次减 dt·u）',
        'execution_mode': {
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'torch.use_deterministic_algorithms': 'True(warn_only=False)',
            'cudnn_benchmark': False, 'cudnn_deterministic': True,
            'cudnn_allow_tf32': False, 'matmul_allow_tf32': False,
            'torch.jit.optimized_execution': False},
    }
    numeric_recipe['module_dtypes'] = out['module_dtypes']
    recipe = dict(numeric_recipe)
    recipe.update({'schema': 'EXPORT01R-models-export/1', 'model_file': 'your_model.pt',
                   'precision_route': ('FULL_FP32_FALLBACK' if full_fp32 else
                                       ('FP32_VAE' if vae_fp32 else 'FP16_REFERENCE')),
                   'notes': ['LoRA 在 fp32 下 merge_and_unload；本文件为其 %s 重载形态' % _wd,
                             'TimestepEmbedder 的 mlp 前插入 CastIn(%s)（cos/sin 仍 fp32 计算，'
                             '全 FP32 下为恒等）' % _wd,
                             'wavelet 为忠实张量化实现，固定 fp32',
                             '噪声显式传入，模块内无随机算子',
                             '分块/融合/补齐裁回在 runner，数值主体在本 ScriptModule']})
    (model_dir / 'models_export.json').write_text(json.dumps(recipe, indent=1, ensure_ascii=False),
                                                  encoding='utf-8')

    bid = bundle_id_of(parents, numeric_recipe)
    runtime_sha = {'chain.py': sha256_file(Path(here) / 'chain.py'),
                   'delivery_runner.py': sha256_file(Path(here) / 'delivery_runner.py'),
                   'r_build.py': sha256_file(Path(here) / 'r_build.py'),
                   'export_config.json': sha256_file(Path(here) / 'export_config.json')}
    bundle_manifest = {
        'schema_version': 'EXPORT01R-bundle-manifest/1',
        'bundle_id': bid,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'parents': parents,
        'conversion': {'id': ('EXPORT01R-fullfp32-jit-v1' if full_fp32
                              else 'EXPORT01R-fp16-jit-v1'), 'status': 'DONE',
                       'method': 'merge_and_unload(fp32) -> CastIn(%s) on t_embedder.mlp/'
                                 'r_embedder.mlp -> module.to(%s) -> torch.jit.trace 子模块 -> '
                                 'torch.jit.script 容器 -> torch.jit.save' % (_wd, _wd),
                       'script_sha256': sha256_file(Path(here) / 'r_build.py'),
                       'env': PBcore.env_fingerprint(),
                       'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'parent_sha256': {p['role']: p.get('sha256') for p in parents},
                       'output_fingerprint': {'file': 'your_model.pt',
                                              'sha256': out['model_sha256'],
                                              'bytes': out['model_bytes']},
                       'cast_wrapping': patched,
                       'device_binding_fix': device_patched,
                       'precision_route': ('FULL_FP32_FALLBACK' if full_fp32 else
                                           ('FP32_VAE' if vae_fp32 else 'FP16_REFERENCE')),
                       'structural_changes': [
                           't_embedder.mlp / r_embedder.mlp 入口 CastIn(%s)' % _wd,
                           'VAE 编码 512/1024 两种 tile trace（同权重同链）',
                           'LightDecoder 64/128 两种 latent tile trace',
                           'wavelet trace 于 64x64，形状通用']},
        'files': [{'path': 'your_model.pt', 'role': 'jit_model', 'bytes': out['model_bytes'],
                   'sha256': out['model_sha256']},
                  {'path': 'runner.py', 'role': 'delivery_runner',
                   'bytes': (model_dir / 'runner.py').stat().st_size,
                   'sha256': sha256_file(model_dir / 'runner.py')},
                  {'path': 'models_export.json', 'role': 'recipe',
                   'bytes': (model_dir / 'models_export.json').stat().st_size,
                   'sha256': sha256_file(model_dir / 'models_export.json')}],
        'numeric_recipe': numeric_recipe,
        'reload_verification': {'status': 'PENDING', 'protocol': 'r_verify.run_verification'},
        'model_dtypes': out['module_dtypes'],
        'runtime_source_sha256': runtime_sha,
    }
    (exp_dir / 'bundle_manifest.json').write_text(
        json.dumps(bundle_manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    out['bundle_id'] = bid

    # ── EXP-R13：保存后必须由"全新进程"复测组件；这里先算并落盘 eager 参考 ──
    import gc
    import r3_component_gate
    out['component_refs'] = r3_component_gate.capture_references(
        (wrap_enc, wrap_dec, wrap_dino, wrap_dit, wrap_wav), dev,
        exp_dir / 'component_refs', ledger, work_dtype)
    del wrap_enc, wrap_dec, wrap_dino, wrap_dit, wrap_wav, merged, vae, dec, dino, scripted
    gc.collect()
    torch.cuda.empty_cache()
    out['build_released'] = True
    out['vram_after_build_release_mib'] = round(torch.cuda.memory_allocated() / 1048576, 1)
    out['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (exp_dir / 'build_result.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                               encoding='utf-8')
    return out
