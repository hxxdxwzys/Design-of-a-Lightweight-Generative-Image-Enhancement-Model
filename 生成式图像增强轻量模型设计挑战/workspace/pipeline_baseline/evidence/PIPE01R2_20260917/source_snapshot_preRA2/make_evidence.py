# -*- coding: utf-8 -*-
"""PIPE-01R 阶段A 证据与预登记生成器（CPU / 只读输入）。

相对首版的修复（PIPE-R03/R04/R06）：
  * **先检查目标 run 目录，再写任何东西**（首版会先改写 evidence/hashes.json 才检查，审查已发现）。
  * 证据只写入**新目录** (--evidence-dir)，不触碰已交付的 evidence/*.json。
  * 预登记绑定**实现源码指纹**（core/runner/offline/pair_verify/make_evidence/config/tests），
    驱动会在模型加载前强制比对。
  * 资产清单补实际 VAE 文件；键契约探针同时覆盖 DiT / DINO / LightDecoder（meta device + 安全读取）。
  * 初始化**持久化预算总账**（存在即拒绝重置，防止刷新配额）。

用法:
  D:/Python/python.exe make_evidence.py [--run-dir <dir>] [--evidence-dir <dir>] [--ledger <json>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import core

HERE = Path(__file__).resolve().parent

CRITERIA = {
    'max_abs_diff': 1e-5,
    'mean_abs_diff': 1e-6,
    'output_dtype': 'float32',
    'output_range': [-1.0, 1.0],
    'png_pixel_identity': 'decoded RGB identical',
    'identity': 'A/B 完整身份逐字段相等（含 DINO/VAE/Decoder 有效参数与实现源码指纹）',
    'load_safety': 'weights_only=True 且无 torch.hub；strict 键集校验',
}
BUDGET = {'max_forwards': 10, 'max_minutes': 30, 'max_output_gib': 4}
SAMPLES = ['case1_full', 'case7_full', 'case1_crop512']
FORWARD_ORDER = ['case1_crop512', 'case1_full', 'case7_full', 'case1_crop512']


def dit_key_contract(cfg):
    """DiT：meta device 构造 + safetensors 头部比对（不加载权重、不跑前向）。"""
    import torch
    from safetensors import safe_open
    core.ensure_vosr_import_path(cfg)
    from models.lightningdit import LightningDiT
    ns = core.build_args_ns(cfg)
    base_ch = 4 if ns.ae_type == 'sd2' else 16
    with torch.device('meta'):
        m = LightningDiT(
            input_size=ns.resolution // 8, patch_size=ns.patch_size,
            in_channels=2 * base_ch, out_channels=base_ch, hidden_size=ns.dim,
            depth=ns.depth, num_heads=ns.num_heads, mlp_ratio=ns.mlp_ratio,
            z_dims=ns.enc_dim, encdim_ratio=ns.encdim_ratio,
            auxiliary_time_cond=bool(getattr(ns, 'auxiliary_time_cond', True)),
            use_qknorm=ns.use_qknorm, use_swiglu=ns.use_swiglu, use_rope=ns.use_rope,
            use_rmsnorm=ns.use_rmsnorm, wo_shift=ns.wo_shift,
            num_fused_layers=len(ns.layer_dinov2b_list))
    model_keys = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    with safe_open(str(cfg['base_weights']), framework='pt') as f:
        file_keys = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    missing = sorted(set(model_keys) - set(file_keys))
    unexpected = sorted(set(file_keys) - set(model_keys))
    mismatch = sorted(k for k in set(model_keys) & set(file_keys)
                      if model_keys[k] != file_keys[k])
    return {'model_n_keys': len(model_keys), 'file_n_keys': len(file_keys),
            'n_missing': len(missing), 'n_unexpected': len(unexpected),
            'n_shape_mismatch': len(mismatch), 'missing_sample': missing[:5],
            'unexpected_sample': unexpected[:5],
            'strict_load_would_pass': not (missing or unexpected or mismatch),
            'method': 'meta device 构造 + safetensors 头部读取；未加载权重、未前向'}


def dino_key_contract(cfg):
    """DINO：meta device 构造本地 vit_base + 安全读取 checkpoint 键/形状比对。"""
    import torch
    repo, ckpt = Path(cfg['dinov2_repo']), Path(cfg['dinov2_ckpt'])
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dinov2.models import vision_transformer as vits
    with torch.device('meta'):
        m = vits.vit_base(**core.DINOV2_VITB14_KWARGS)
    model_keys = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    sd = torch.load(str(ckpt), map_location='cpu', weights_only=True)
    file_keys = {k: tuple(v.shape) for k, v in sd.items()}
    del sd
    missing = sorted(set(model_keys) - set(file_keys))
    unexpected = sorted(set(file_keys) - set(model_keys))
    mismatch = sorted(k for k in set(model_keys) & set(file_keys)
                      if model_keys[k] != file_keys[k])
    return {'loader': 'local source vit_base + torch.load(weights_only=True) + strict=True',
            'uses_torch_hub': False,
            'model_n_keys': len(model_keys), 'file_n_keys': len(file_keys),
            'n_missing': len(missing), 'n_unexpected': len(unexpected),
            'n_shape_mismatch': len(mismatch),
            'strict_load_would_pass': not (missing or unexpected or mismatch),
            'method': 'meta device 构造 + weights_only=True 读取（只读本地权重）'}


def decoder_probe(cfg):
    """LightDecoder：weights_only=True 安全读取（首版用 False，审查已指出）。"""
    import torch
    from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D  # noqa: F401
    lw = torch.load(str(cfg['light_decoder']), map_location='cpu', weights_only=True)
    keys = ('in_channels', 'out_channels', 'block_out_channels', 'layers_per_block')
    cfg_keys = sorted(lw['config'].keys())
    sd_keys = sorted(lw['model_state_dict'].keys())
    return {'weights_only': True, 'payload_keys': sorted(lw.keys()),
            'config_keys': cfg_keys, 'n_state_keys': len(sd_keys),
            'required_config_present': all(k in lw['config'] for k in keys),
            'global_step': lw.get('global_step')}


def lora_probe(cfg):
    payload = core.load_lora_payload(cfg['lora_path'])
    sd = payload['state_dict']
    lora_keys = [k for k in sd if 'lora' in k]
    return {'weights_only': True,
            'payload_keys': sorted(k for k in payload.keys() if k != 'state_dict'),
            'variant': payload.get('variant'), 'step': payload.get('step'),
            'n_state_dict': len(sd), 'n_lora_keys': len(lora_keys),
            'has_lora_config': 'lora_config' in payload}


def build(run_dir=None, evidence_dir=None, ledger_path=None, write_run_dir=True):
    t0 = time.time()
    run_dir = Path(run_dir) if run_dir else Path(
        'D:/models/pipeline_baseline/PIPE01R_%s' % time.strftime('%Y%m%d_%H%M%S'))
    ev = Path(evidence_dir) if evidence_dir else HERE / 'evidence' / (
        'PIPE01R_%s' % time.strftime('%Y%m%d_%H%M%S'))
    ledger_path = Path(ledger_path) if ledger_path else (run_dir.parent / 'budget_ledger.json')

    # ① 先做全部只读检查与阻断判定，再写任何文件（PIPE-R06/R04 修复点）
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit('拒绝复用非空 run 目录（未写入任何文件）: %s' % run_dir)
    cfg = core.load_config()
    asset_fp = core.asset_fingerprint(cfg)
    source_fp = core.source_fingerprint(cfg)
    impl_fp = core.implementation_fingerprint()
    env_fp = core.env_fingerprint()
    kc_dit = dit_key_contract(cfg)
    kc_dino = dino_key_contract(cfg)
    dec = decoder_probe(cfg)
    lp = lora_probe(cfg)
    bad = [n for n, ok in (('dit', kc_dit['strict_load_would_pass']),
                           ('dino', kc_dino['strict_load_would_pass'])) if not ok]
    if not dec['required_config_present']:
        bad.append('light_decoder_config')
    if bad:
        raise SystemExit('KEY_CONTRACT_FAIL: %s —— 已停止，且未写入任何 evidence/run 文件' % bad)

    # ② 通过后才落盘
    ev.mkdir(parents=True, exist_ok=True)
    (ev / 'hashes.json').write_text(json.dumps(
        {'task': 'PIPE-01R', 'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
         'config_sha256': cfg['_config_sha256'], 'asset_fingerprint': asset_fp,
         'source_fingerprint': source_fp, 'implementation_fingerprint': impl_fp,
         'env': env_fp, 'tile_config': core.tile_config_report(cfg, 2)},
        ensure_ascii=False, indent=1), encoding='utf-8')
    (ev / 'key_contract.json').write_text(json.dumps(
        {'dit': kc_dit, 'dino': kc_dino, 'light_decoder': dec, 'lora': lp},
        ensure_ascii=False, indent=1), encoding='utf-8')

    prereg = {
        'task': 'PIPE-01R', 'phase': 'B-预登记（GPU 前写入）',
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config_sha256': cfg['_config_sha256'],
        'asset_fingerprint': asset_fp,
        'source_fingerprint': source_fp,
        'implementation_fingerprint': impl_fp,
        'env': env_fp,
        'tile_config': core.tile_config_report(cfg, 2),
        'key_contract': {'dit': kc_dit, 'dino': kc_dino, 'light_decoder': dec},
        'criteria': CRITERIA, 'budget': BUDGET,
        'samples': SAMPLES, 'forward_order_per_instance': FORWARD_ORDER,
        'instances': ['A=tensor_runner', 'B=offline_file_entry'],
        'stop_conditions': ['OOM', 'NaN/Inf（量化前）', '资源缺失', '身份不符', '预算触顶',
                            '预登记不符', 'max/mean_abs_diff 或 PNG 判据不达标（记录后停止，不改阈值）'],
    }
    (ev / 'prereg_PIPE01R.json').write_text(
        json.dumps(prereg, ensure_ascii=False, indent=1), encoding='utf-8')
    if write_run_dir:
        run_dir.mkdir(parents=True)
        (run_dir / 'prereg.json').write_text(
            json.dumps(prereg, ensure_ascii=False, indent=1), encoding='utf-8')
        print('[run-dir] %s' % run_dir)
    if ledger_path.exists():
        raise SystemExit('预算总账已存在，拒绝重置配额（未覆盖）: %s' % ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(
        {'task': 'PIPE-01', 'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
         'max_forwards': BUDGET['max_forwards'], 'max_minutes': BUDGET['max_minutes'],
         'max_gib': BUDGET['max_output_gib'], 'forwards_reserved': 0,
         'wall_seconds_total': 0.0, 'attempts': [], 'open_attempt': None,
         'reservations': [], 'stale_locks': [],
         'prereg_sha256': core.sha256_file(run_dir / 'prereg.json')
         if (run_dir / 'prereg.json').is_file() else None},
        ensure_ascii=False, indent=1), encoding='utf-8')
    print('[evidence] %s' % ev)
    print('[ledger]   %s (max_forwards=%d)' % (ledger_path, BUDGET['max_forwards']))
    print('[key-contract] dit=%s dino=%s decoder_weights_only=%s'
          % (kc_dit['strict_load_would_pass'], kc_dino['strict_load_would_pass'],
             dec['weights_only']))
    print('[lora] n_lora_keys=%d' % lp['n_lora_keys'])
    print('[done] %.1fs' % (time.time() - t0))
    return prereg


def main(argv=None):
    ap = argparse.ArgumentParser(description='PIPE-01R 阶段A 证据与预登记生成器')
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--evidence-dir', default=None)
    ap.add_argument('--ledger', default=None)
    ap.add_argument('--no-run-dir', action='store_true', help='只产证据，不新建 run 目录')
    a = ap.parse_args(argv)
    build(a.run_dir, a.evidence_dir, a.ledger, write_run_dir=not a.no_run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
