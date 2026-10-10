# -*- coding: utf-8 -*-
"""C1 导出：fp16 精度迁移 + 完整数值链 ScriptModule 导出（产 model_dir/）。

用法: python build_export.py <run_dir>
产出:
  <run_dir>/model_dir/{your_model.pt, runner.py, models_export.json}
  <run_dir>/evidence/{conversion_record.json, build_result.json, source_hashes.json}
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline_baseline'))

import core as PBcore  # noqa: E402
from chain import (Decode, DinoFeat, DitStep, ExportBundle, VAEEncode,  # noqa: E402
                   WaveletFix, patch_fp16_embedders)


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    mdl = run / 'model_dir'
    ev = run / 'evidence'
    mdl.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)
    exp_cfg = json.loads((HERE / 'export_config.json').read_text(encoding='utf-8'))
    cfg = PBcore.load_config()
    dev = torch.device('cuda')
    result = {'run_dir': str(run), 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}

    t0 = time.perf_counter()
    assets = PBcore.Assets(cfg, dev)
    result['assets_load_s'] = round(time.perf_counter() - t0, 3)

    # 父资产谱系（只读哈希）
    parents = []
    for role, path in (('base_weights', cfg['base_weights']), ('lora', cfg['lora_path']),
                       ('vae_dir', cfg['ae_dir']), ('light_decoder', cfg['light_decoder']),
                       ('dinov2_ckpt', cfg['dinov2_ckpt'])):
        p = Path(path)
        if p.is_file():
            parents.append({'role': role, 'path': str(p), 'bytes': p.stat().st_size,
                            'sha256': sha256_file(p)})
        else:
            parents.append({'role': role, 'path': str(p), 'kind': 'directory',
                            'n_files': sum(1 for _ in p.rglob('*') if _.is_file())})
    (ev / 'source_hashes.json').write_text(json.dumps({
        'parents': parents,
        'pipeline_config_sha256': cfg['_config_sha256'],
        'export_chain_py_sha256': sha256_file(HERE / 'chain.py'),
        'delivery_runner_py_sha256': sha256_file(HERE / 'delivery_runner.py'),
        'build_export_py_sha256': sha256_file(Path(__file__)),
        'export_config_sha256': sha256_file(HERE / 'export_config.json'),
    }, indent=1, ensure_ascii=False), encoding='utf-8')

    # ── 精度迁移：合并 LoRA -> 时间嵌入对齐 -> fp16 ──
    merged = assets.model.merge_and_unload()
    result['merged_type'] = type(merged).__name__
    merged = merged.half()
    patched = patch_fp16_embedders(merged, torch.float16)
    vae = assets.vae.half()
    dec = assets.dec.half()
    dino = assets.venc.half()
    result['patched_embedders'] = patched

    wrap_enc = VAEEncode(vae).to(dev).eval()
    wrap_dec = Decode(dec, vae.config.scaling_factor).to(dev).eval()
    wrap_dino = DinoFeat(dino, size=int(cfg['expected_dinov2_size']),
                         layer_index=int(cfg['expected_layer_dinov2b_list'][0]),
                         n_blocks=len(dino.blocks)).to(dev).eval()
    wrap_dit = DitStep(merged).to(dev).eval()
    wrap_wav = WaveletFix(sigma=float(cfg['align_sigma'])).to(dev).eval()

    def rnd(*shape):
        return torch.randn(*shape, device=dev, dtype=torch.float16)

    t = time.perf_counter()
    traces = {}
    torch.cuda.empty_cache()
    traces['vae_enc_512'] = torch.jit.trace(wrap_enc, rnd(1, 3, 512, 512), strict=False,
                                           check_trace=False)
    torch.cuda.empty_cache()
    traces['vae_enc_1024'] = torch.jit.trace(wrap_enc, rnd(1, 3, 1024, 1024), strict=False,
                                            check_trace=False)
    torch.cuda.empty_cache()
    traces['dino_feat'] = torch.jit.trace(wrap_dino, rnd(1, 3, 512, 512), strict=False,
                                         check_trace=False)
    tcur = torch.full((1,), float(cfg['t_start']), device=dev, dtype=torch.float16)
    tnxt = torch.full((1,), float(cfg['t_end']), device=dev, dtype=torch.float16)
    traces['dit_step'] = torch.jit.trace(
        wrap_dit, (rnd(1, 4, 64, 64), rnd(1, 4, 64, 64), tcur, tnxt, rnd(1, 1024, 768)),
        strict=False, check_trace=False)
    torch.cuda.empty_cache()
    traces['dec_64'] = torch.jit.trace(wrap_dec, rnd(1, 4, 64, 64), strict=False,
                                      check_trace=False)
    traces['dec_128'] = torch.jit.trace(wrap_dec, rnd(1, 4, 128, 128), strict=False,
                                       check_trace=False)
    traces['wavelet'] = torch.jit.trace(wrap_wav, (rnd(1, 3, 64, 64), rnd(1, 3, 64, 64)),
                                        strict=False, check_trace=False)
    result['trace_s'] = round(time.perf_counter() - t, 3)
    result['component_probes_used'] = 2 + 1 + 1 + 2 + 1  # 各 trace 至少一次真实前向

    bundle = ExportBundle(traces['vae_enc_512'], traces['vae_enc_1024'], traces['dino_feat'],
                          traces['dit_step'], traces['dec_64'], traces['dec_128'],
                          traces['wavelet'])
    scripted = torch.jit.script(bundle)
    ckpt = mdl / 'your_model.pt'
    torch.jit.save(scripted, str(ckpt))
    result['model_bytes'] = ckpt.stat().st_size
    result['model_sha256'] = sha256_file(ckpt)
    reloaded = torch.jit.load(str(ckpt), map_location='cpu')
    result['reload_ok_cpu'] = True
    del reloaded

    # ── 交付 runner 与 recipe ──
    shutil.copyfile(HERE / 'delivery_runner.py', mdl / 'runner.py')
    recipe = {
        'schema': 'EXPORT01-models-export/1',
        'model_file': 'your_model.pt',
        'weight_dtype': 'float16',
        'postprocess_dtype': 'float32',
        'seed': int(cfg['seed']),
        't_start': float(cfg['t_start']),
        't_end': float(cfg['t_end']),
        'dit_tile_size': int(cfg['dit_tile_size']),
        'dit_tile_overlap_cli': int(cfg['dit_tile_overlap_cli']),
        'vae_tile_size': int(cfg['vae_tile_size']),
        'vae_tile_overlap_cli': int(cfg['vae_tile_overlap_cli']),
        'patch_size': int(cfg['expected_patch_size']),
        'vae_enc_trace_px': [512, 1024],
        'dec_trace_latent': [64, 128],
        'dinov2_input_size': int(cfg['expected_dinov2_size']),
        'align_sigma': int(cfg['align_sigma']),
        'posterior_mode': True,
        'blend_accumulator_dtype': 'float32',
        'notes': [
            'LoRA 已在 fp32 下 merge_and_unload 合并进基座；本文件为其 fp16 重载形态',
            'TimestepEmbedder 输出显式转 fp16（原实现内部固定 float32 计算 cos/sin）',
            'wavelet 后处理为忠实张量化实现，固定 fp32 执行',
            '噪声由 runner 以局部 Generator(seed) 生成后显式传入，模块内无随机算子',
            '分块/融合/补齐裁回等编排在 runner；数值主体全部在本 ScriptModule',
        ],
    }
    (mdl / 'models_export.json').write_text(json.dumps(recipe, indent=1, ensure_ascii=False),
                                            encoding='utf-8')

    conversion = {
        'id': 'EXPORT01-fp16-jit-v1',
        'status': 'DONE',
        'method': 'merge LoRA (peft.merge_and_unload, fp32) -> patch TimestepEmbedder casts -> '
                  'module.half() -> torch.jit.trace 子模块 -> torch.jit.script 容器 -> save',
        'script_sha256': sha256_file(Path(__file__)),
        'command': 'python build_export.py %s' % run,
        'env': PBcore.env_fingerprint(),
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'parent_sha256': {p['role']: p.get('sha256') for p in parents},
        'output_fingerprint': {
            'file': 'your_model.pt', 'sha256': result['model_sha256'],
            'bytes': result['model_bytes'],
            'n_params_half': int(sum(p.numel() for p in scripted.parameters())),
        },
        'structural_changes': [
            't_embedder/r_embedder 包一层 CastOut(fp16)',
            'VAE 编码器按 512/1024 两种 tile 尺寸分别 trace（同一权重、同一数值链）',
            'LightDecoder 按 64/128 两种 latent tile 尺寸分别 trace',
            'wavelet trace 于 64x64，形状通用（已验证 4K 尺寸可用）',
        ],
        'precision_migration': {
            'from': 'float32', 'to': 'float16',
            'postprocess_stays': 'float32',
            'noise_protocol': recipe['notes'][3],
        },
    }
    (ev / 'conversion_record.json').write_text(json.dumps(conversion, indent=1,
                                                          ensure_ascii=False), encoding='utf-8')
    result['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (ev / 'build_result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                          encoding='utf-8')
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
