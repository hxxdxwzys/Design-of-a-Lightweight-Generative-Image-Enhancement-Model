# -*- coding: utf-8 -*-
"""EXPORT-01R2 准备（CPU，先于 GPU attempt）：
  - 只读校验旧 .pt（sha256/字节数必须匹配裁决给定值）；
  - 复制到新 run 的 model_dir，前后字节校验一致（禁止硬链接后改写）；
  - 写入**修正后的交付 runner**（含冻结 JIT 执行模式与 EXP-R08 计算顺序）与 recipe；
  - 形成**新 bundle 身份**（同一 .pt 复用 + runner 修复 + 新协议），不沿用旧 bundle_id。

用法: python r2_prepare.py <new_run_dir> <new_evidence_dir>
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_PT = Path('C:/models/exports/EXPORT01R_20260917_150506/model_dir/your_model.pt')
OLD_LEDGER = Path('C:/models/exports/EXPORT01R_20260917_150506/ledger.json')
EXPECT_SHA = '7b8152029e3d96cf90e0787b42949e680cdb3c1bdfce6f6be99452444283ccc6'
EXPECT_BYTES = 1343344555


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def bundle_id_of(payload):
    return 'EXPORT01R2-' + hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:32]


def main():
    run = Path(sys.argv[1]).resolve()
    ev = Path(sys.argv[2]).resolve()
    mdl = run / 'model_dir'
    mdl.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)
    rec = {'schema': 'EXPORT01R2-prepare/1', 'at': time.strftime('%Y-%m-%d %H:%M:%S')}

    src_sha = sha256_file(OLD_PT)
    src_bytes = OLD_PT.stat().st_size
    if src_sha != EXPECT_SHA or src_bytes != EXPECT_BYTES:
        raise SystemExit('旧模型身份不符：sha=%s bytes=%d' % (src_sha, src_bytes))
    rec['old_model'] = {'path': str(OLD_PT), 'sha256': src_sha, 'bytes': src_bytes,
                        'verified_against_ruling': True}
    dst = mdl / 'your_model.pt'
    shutil.copyfile(OLD_PT, dst)
    dst_sha, dst_bytes = sha256_file(dst), dst.stat().st_size
    if (dst_sha, dst_bytes) != (src_sha, src_bytes):
        raise SystemExit('复制后字节不一致')
    rec['copied_model'] = {'path': str(dst), 'sha256': dst_sha, 'bytes': dst_bytes,
                           'byte_identical': True, 'hardlink': False}

    shutil.copyfile(HERE / 'delivery_runner.py', mdl / 'runner.py')
    old_recipe = json.loads((OLD_PT.parent / 'models_export.json').read_text(encoding='utf-8'))
    recipe = dict(old_recipe)
    recipe['runtime'] = {'jit_optimized_execution': False,
                         'deterministic_algorithms': 'True(warn_only=False)',
                         'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                         'cudnn_benchmark': False, 'cudnn_deterministic': True,
                         'cudnn_allow_tf32': False, 'matmul_allow_tf32': False}
    recipe['fusion_and_order'] = {
        'step_semantics': 'DitStep 返回更新后的 z1；分块融合 z1（EXP-R01）',
        'decode_order': '逐块 raw decode -> finite 检查 -> clamp -> 加权融合 -> 最终 finite 检查 -> 最终 clamp（EXP-R08）'}
    recipe['notes'] = list(recipe.get('notes', [])) + [
        'EXPORT-01R2：模型文件与 EXPORT-01R 转换产物逐字节相同；runner 为修正版（新 bundle 身份）']
    (mdl / 'models_export.json').write_text(json.dumps(recipe, indent=1, ensure_ascii=False),
                                            encoding='utf-8')

    old_ledger_sha = sha256_file(OLD_LEDGER) if OLD_LEDGER.is_file() else None
    old_ledger = json.loads(OLD_LEDGER.read_text(encoding='utf-8')) if OLD_LEDGER.is_file() else {}
    parents = [{'role': 'model_from_EXPORT01R', 'sha256': src_sha, 'bytes': src_bytes,
                'reused': True, 'note': '同一转换产物，未重建；数值验收不因复用而通过'}]
    runtime_sha = {'chain.py': sha256_file(HERE / 'chain.py'),
                   'delivery_runner.py': sha256_file(HERE / 'delivery_runner.py'),
                   'r_verify.py': sha256_file(HERE / 'r_verify.py'),
                   'r_ledger.py': sha256_file(HERE / 'r_ledger.py'),
                   'r_merge_probe.py': sha256_file(HERE / 'r_merge_probe.py'),
                   'r_supervisor.py': sha256_file(HERE / 'r_supervisor.py')}
    numeric_recipe = {'weight_dtype': recipe['weight_dtype'], 'seed': recipe['seed'],
                      't_start': recipe['t_start'], 't_end': recipe['t_end'],
                      'dit_tile_size': recipe['dit_tile_size'],
                      'dit_tile_overlap_cli': recipe['dit_tile_overlap_cli'],
                      'vae_tile_size': recipe['vae_tile_size'],
                      'vae_tile_overlap_cli': recipe['vae_tile_overlap_cli'],
                      'patch_size': recipe['patch_size'],
                      'execution_mode': recipe['runtime'],
                      'fusion_and_order': recipe['fusion_and_order']}
    bid = bundle_id_of({'model_sha256': src_sha, 'runtime_sha256': runtime_sha,
                        'numeric_recipe': numeric_recipe, 'batch': 'EXPORT01R2'})
    manifest = {'schema_version': 'EXPORT01R2-bundle-manifest/1', 'bundle_id': bid,
                'created_at': rec['at'],
                'model_reuse': {'from_batch': 'EXPORT-01R', 'hardlink': False,
                                'byte_identical': True,
                                'old_bundle_note': '旧 bundle_id 不复用：runner 与协议已变更'},
                'parents': parents,
                'conversion': {'id': 'EXPORT01R2-runner-fix-v1', 'status': 'RUNNER_ONLY_FIX',
                               'note': '数值图未重建（未授权）；仅交付 runner 修复 + 新协议',
                               'script_sha256': sha256_file(Path(__file__))},
                'files': [{'path': 'your_model.pt', 'role': 'jit_model', 'sha256': src_sha,
                           'bytes': src_bytes},
                          {'path': 'runner.py', 'role': 'delivery_runner',
                           'sha256': runtime_sha['delivery_runner.py'],
                           'bytes': (mdl / 'runner.py').stat().st_size},
                          {'path': 'models_export.json', 'role': 'recipe',
                           'sha256': sha256_file(mdl / 'models_export.json')}],
                'numeric_recipe': numeric_recipe,
                'runtime_source_sha256': runtime_sha,
                'prior_batch': {'ledger': str(OLD_LEDGER), 'ledger_sha256': old_ledger_sha,
                                'component_used': old_ledger.get('used', {}).get('component'),
                                'supervisor_wall_s': 71.083,
                                'note': '旧失败与扣款保留；新批 component 上限 38，旧+新≤48'}}
    (ev / 'bundle_manifest_r2.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                                                encoding='utf-8')
    rec['bundle_id'] = bid
    rec['new_bundle_manifest'] = str(ev / 'bundle_manifest_r2.json')
    (ev / 'prepare_r2.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                                        encoding='utf-8')
    print(json.dumps({'bundle_id': bid, 'model_sha256': dst_sha, 'model_bytes': dst_bytes,
                      'old_ledger_sha256': old_ledger_sha}, ensure_ascii=False))


if __name__ == '__main__':
    main()
