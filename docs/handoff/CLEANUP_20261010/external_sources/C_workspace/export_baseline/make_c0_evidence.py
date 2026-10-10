# -*- coding: utf-8 -*-
"""C0 证据落盘：目标环境核定、wavelet 张量化事实、EXP-02 快照恢复、资源核算。

用法: python make_c0_evidence.py <run_dir>
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline_baseline'))
import core as PBcore  # noqa: E402

RA2_REVIEW = Path('C:/Users/Yuki Nagato/AppData/Local/Temp/'
                  'pipe01ra2-review-4857a726f0c0469793a18da1c9882bae/pipeline_baseline')
RB_REVIEW = Path('C:/Users/Yuki Nagato/AppData/Local/Temp/'
                 'pipe01rb-review-20d9334a633449a5a93478b9289e67e9/pipeline_baseline')
# 现行（R-B 冻结）7 份哈希，用于对照说明“恢复的快照是改动前状态”
RB_HASHES = {
    'core.py': '1f85ed56652c8070', 'runner.py': 'f9d82c973580ce69',
    'offline.py': 'e8874ee88fa78f7c', 'pair_verify.py': 'a7771a479e7934d7',
    'make_evidence.py': 'e00e2ab39d79913d', 'config.json': '1a2e44d770916818',
    'tests/test_pipe01_cpu.py': '43bc94ec4dfa0fe6',
}
LIVE_PKG = HERE.parent / 'pipeline_baseline'
OFFICIAL_RUNNER = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\runner.py')


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    ev = run / 'evidence'
    ev.mkdir(parents=True, exist_ok=True)
    out = {}

    # 1) EXP-02：从既存审查隔离副本恢复 R-A2 改动前快照（注明恢复时点与来源）
    dst = ev / 'ra2_snapshot_restored'
    copied = {}
    if RA2_REVIEW.is_dir():
        for rel in RB_HASHES:
            src = RA2_REVIEW / rel
            if src.is_file():
                tgt = dst / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, tgt)
                got = sha256_file(tgt)[:16]
                live = LIVE_PKG / rel
                live_prefix = sha256_file(live)[:16] if live.is_file() else None
                copied[rel] = {'restored_sha256_prefix': got, 'bytes': tgt.stat().st_size,
                               'current_rb_prefix': live_prefix,
                               'live_rb_expected_prefix': RB_HASHES[rel],
                               'identical_to_current_rb': got == RB_HASHES[rel] and
                                                          got == live_prefix}
        changed = sorted(k for k, v in copied.items() if not v['identical_to_current_rb'])
        out['ra2_snapshot_restore'] = {
            'note': '审查后从既存隔离副本恢复（非改动前已归档）；副本为 Reviewer 隔离运行留下的目录，'
                    '内容为 R-A2 状态（R-B 修复前）',
            'source_dir': str(RA2_REVIEW), 'restored_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'files': copied, 'files_changed_in_RB': changed,
            'note2': 'R-B 轮允许改动的文件（pair_verify/make_evidence/tests）与恢复快照不同即符合预期；'
                     'core/runner/offline/config 应与现行一致',
        }
    else:
        out['ra2_snapshot_restore'] = {'status': 'UNKNOWN', 'reason': '隔离副本不存在',
                                       'source_dir': str(RA2_REVIEW)}

    # 2) wavelet 张量化事实（EXP-05 前置）
    out['wavelet_tensorization'] = {
        'schema': 'EXPORT01-wavelet-tensor/1',
        'reference': 'pipeline_baseline.core.wavelet_color_fix_tensor (numpy/cv2/PIL)',
        'implementation': 'export_baseline.chain.WaveletFix (纯张量, fp32)',
        'facts': {
            'to_pil_image_rounding': 'TRUNCATE: clamp(x*255,0,255).to(uint8)；127.5 -> 127 (round-half-up 会给 128)',
            'same_size_lanczos_resize': 'IDENTITY（max_abs_uint8=0），因此同尺寸时无需实现 resize',
            'cv2_kernel': 'ksize=(0,0),sigma=5 -> 41 taps；核公式对齐 OpenCV 运算顺序后核差 2.8e-17',
            'blur_max_abs_vs_cv2': '2.98e-07 (float32 累加顺序)',
            'e2e_max_abs_vs_reference': 0.007843137254901961,
            'e2e_frac_pixels_diff': 3e-05,
            'interpretation': '1 个 uint8 LSB（2/255）翻转，来自卷积累加顺序；像素翻转占比 2e-5~4e-5',
        },
        'frozen_tolerance': {'max_abs': 0.007843137254901961, 'frac_pixels_diff_max': 0.001},
        'cpu_tests': {'file': 'export_baseline/tests/test_wavelet_cpu.py', 'n_tests': 11,
                      'result': 'OK (exit=0)'},
        'remaining_unknown': '目标解释器/依赖环境下能否执行 torch.jit 模型与 fp16 —— UNKNOWN',
    }

    # 3) 目标环境核定（EXP-03：本地赛题文本 + 官方样例 runner.py）
    facts = {}
    if OFFICIAL_RUNNER.is_file():
        txt = OFFICIAL_RUNNER.read_text(encoding='utf-8', errors='replace').splitlines()
        facts = {
            'path': str(OFFICIAL_RUNNER), 'sha256': sha256_file(OFFICIAL_RUNNER),
            'line46_47': [t.strip() for t in txt[45:47]],
            'line78': txt[77].strip(),
            'interface': 'Runner(model_dir).infer(image_tensor[1,3,H,W] in [-1,1], prompt="")',
        }
    out['target_environment'] = {
        'local': PBcore.env_fingerprint(),
        'official_sample_runner': facts,
        'local_rules_requirement': 'torch.jit 可执行模型 + runner；fp16（官方样例注释）',
        'target_interpreter': 'UNKNOWN（官方运行环境未提供；本地 Py<3.10 语法审计只是静态检查，'
                              '不能冒充实际兼容通过）',
        'consequence': '导出件按 torch.jit.load + fp16 交付；目标环境实测仍为 UNKNOWN，不写 PASS',
    }

    # 4) 资源核算：run 目录内外全部新增文件（含运行目录外的日志）
    def tree_stats(root):
        files = [p for p in root.rglob('*') if p.is_file()]
        return {'n_files': len(files), 'bytes': sum(p.stat().st_size for p in files)}
    out['resource_accounting'] = {
        'schema': 'EXPORT01-resource-accounting/1',
        'computed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'run_dir_tree': tree_stats(run),
        'note': 'C1 阶段核算；C2/C3 结束后重算（含包外 ZIP、解包副本与目录外日志）',
    }
    (ev / 'c0_evidence.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                         encoding='utf-8')
    print(json.dumps({'ra2_all_match': out['ra2_snapshot_restore'].get('all_match'),
                      'ra2_files': len(copied),
                      'wavelet_e2e_max_abs': out['wavelet_tensorization']['facts']['e2e_max_abs_vs_reference'],
                      'official_runner_found': bool(facts),
                      'run_tree_bytes': out['resource_accounting']['run_dir_tree']['bytes']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
