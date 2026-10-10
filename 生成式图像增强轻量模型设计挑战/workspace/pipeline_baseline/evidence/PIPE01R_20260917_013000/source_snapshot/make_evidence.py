# -*- coding: utf-8 -*-
"""PIPE-01 阶段A 证据与预登记生成器（CPU / 只读输入）。

做三件事：
  1. 记录资产/源码身份链与冻结配置（流式哈希，不改动任何被登记文件）。
  2. 用 meta device 做**键契约探针**：不加载权重、不执行前向，只比较
     LightningDiT 的参数名/形状与 safetensors 头部的键/形状，提前暴露加载格式问题。
  3. 全部探针通过后，才新建 run 目录并写入 prereg.json（判据+预算）。
     任何一步失败都会抛错且**不写任何“成功 manifest”**。

用法: D:/Python/python.exe make_evidence.py [--run-dir D:/models/pipeline_baseline/PIPE01_<ts>]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import core

HERE = Path(__file__).resolve().parent
EV = HERE / 'evidence'

CRITERIA = {
    'max_abs_diff': 1e-5,
    'mean_abs_diff': 1e-6,
    'output_dtype': 'float32',
    'output_range': [-1.0, 1.0],
    'png_pixel_identity': 'decoded RGB identical',
}
BUDGET = {'max_forwards': 10, 'max_minutes': 30, 'max_output_gib': 4}
SAMPLES = ['case1_full', 'case7_full', 'case1_crop512']
FORWARD_ORDER = ['case1_crop512', 'case1_full', 'case7_full', 'case1_crop512']


def key_contract(cfg):
    """meta device 键契约：不加载权重、不跑前向。"""
    import torch
    from safetensors import safe_open
    core.ensure_vosr_import_path(cfg)
    from models.lightningdit import LightningDiT
    ns = core.build_args_ns(cfg)
    base_ch = 4 if ns.ae_type == 'sd2' else 16
    with torch.device('meta'):
        m = LightningDiT(
            input_size=ns.resolution // 8, patch_size=ns.patch_size,
            in_channels=2 * base_ch, out_channels=base_ch,
            hidden_size=ns.dim, depth=ns.depth, num_heads=ns.num_heads,
            mlp_ratio=ns.mlp_ratio, z_dims=ns.enc_dim, encdim_ratio=ns.encdim_ratio,
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
    return {
        'model_n_keys': len(model_keys), 'file_n_keys': len(file_keys),
        'missing_in_file': missing[:10], 'n_missing': len(missing),
        'unexpected_in_file': unexpected[:10], 'n_unexpected': len(unexpected),
        'shape_mismatch': mismatch[:10], 'n_shape_mismatch': len(mismatch),
        'strict_load_would_pass': not (missing or unexpected or mismatch),
        'method': 'meta device 构造 + safetensors 头部读取，未加载权重、未执行前向',
    }


def lora_probe(cfg):
    """只验证 LoRA 是否能用 weights_only=True 安全加载（不装载模型）。"""
    payload = core.load_lora_payload(cfg['lora_path'])
    sd = payload['state_dict']
    lora_keys = [k for k in sd if 'lora' in k]
    return {
        'weights_only': True,
        'payload_keys': sorted(k for k in payload.keys() if k != 'state_dict'),
        'variant': payload.get('variant'), 'step': payload.get('step'),
        'n_state_dict': len(sd), 'n_lora_keys': len(lora_keys),
        'lora_key_sample': sorted(lora_keys)[:5],
        'has_lora_config': 'lora_config' in payload,
    }


def build(run_dir=None, write_run_dir=True):
    t0 = time.time()
    cfg = core.load_config()
    EV.mkdir(parents=True, exist_ok=True)
    out = {
        'task': 'PIPE-01', 'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config_path': cfg['_config_path'], 'config_sha256': cfg['_config_sha256'],
        'tile_config': core.tile_config_report(cfg, 2),
        'asset_fingerprint': core.asset_fingerprint(cfg),
        'source_fingerprint': core.source_fingerprint(cfg),
        'env': {'python': sys.version.split()[0], 'platform': platform.platform()},
    }
    (EV / 'hashes.json').write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                    encoding='utf-8')
    kc = key_contract(cfg)
    lp = lora_probe(cfg)
    (EV / 'key_contract.json').write_text(
        json.dumps({'key_contract': kc, 'lora_probe': lp}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    if not kc['strict_load_would_pass']:
        raise SystemExit('KEY_CONTRACT_FAIL: 基座键集与模型不一致，停止（不写 prereg）: %s'
                         % {k: kc[k] for k in ('n_missing', 'n_unexpected', 'n_shape_mismatch')})
    prereg = {
        'task': 'PIPE-01', 'phase': 'B-预登记（GPU 前写入）',
        'generated_at': out['generated_at'],
        'config_sha256': cfg['_config_sha256'],
        'asset_fingerprint': out['asset_fingerprint'],
        'source_fingerprint': out['source_fingerprint'],
        'tile_config': out['tile_config'],
        'key_contract': {k: kc[k] for k in ('strict_load_would_pass', 'model_n_keys',
                                            'file_n_keys')},
        'criteria': CRITERIA, 'budget': BUDGET,
        'samples': SAMPLES, 'forward_order_per_instance': FORWARD_ORDER,
        'instances': ['A=tensor_runner', 'B=offline_file_entry'],
        'stop_conditions': ['OOM', 'NaN/Inf', '资源缺失', '预算触顶',
                            'max/mean_abs_diff 超判据（记录后停止，不改阈值）'],
    }
    if write_run_dir:
        run_dir = Path(run_dir) if run_dir else Path(
            'D:/models/pipeline_baseline/PIPE01_%s' % time.strftime('%Y%m%d_%H%M%S'))
        if run_dir.exists():
            raise SystemExit('拒绝覆盖已存在的 run 目录: %s' % run_dir)
        run_dir.mkdir(parents=True)
        (run_dir / 'prereg.json').write_text(
            json.dumps(prereg, ensure_ascii=False, indent=1), encoding='utf-8')
        print('[run-dir] %s' % run_dir)
    (EV / 'prereg_PIPE01.json').write_text(json.dumps(prereg, ensure_ascii=False, indent=1),
                                           encoding='utf-8')
    print('[evidence] %s' % (EV / 'hashes.json'))
    print('[key-contract] strict_load_would_pass=%s keys=%d' % (kc['strict_load_would_pass'],
                                                                kc['model_n_keys']))
    print('[lora] n_lora_keys=%d payload=%s' % (lp['n_lora_keys'], lp['payload_keys']))
    print('[done] %.1fs' % (time.time() - t0))
    return prereg


def main(argv=None):
    ap = argparse.ArgumentParser(description='PIPE-01 阶段A 证据与预登记生成器')
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--no-run-dir', action='store_true', help='只产证据，不新建 run 目录')
    a = ap.parse_args(argv)
    build(a.run_dir, write_run_dir=not a.no_run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
