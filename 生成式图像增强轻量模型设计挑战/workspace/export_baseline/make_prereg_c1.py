# -*- coding: utf-8 -*-
"""C1 预登记（先登记、后执行）：固定输入、判据、预算、停止条件。

用法: python make_prereg_c1.py <prereg_out.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline_baseline'))

TEST_DIR = Path(r'C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集')


def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1])
    cases = []
    for name, src, box in (
        ('case1_crop512', 'case1.jpg', (1792, 1280, 2304, 1792)),
        ('case1_full', 'case1.jpg', None),
        ('case7_full', 'case7.jpg', None),
        ('case1_crop511x513', 'case1.jpg', (1792, 1280, 1792 + 513, 1280 + 511)),
    ):
        p = TEST_DIR / src
        im = Image.open(p).convert('RGB')
        if box:
            im = im.crop(box)
        arr = np.asarray(im)
        cases.append({
            'name': name, 'source_file': str(p), 'source_sha256': file_sha(p),
            'crop_box': list(box) if box else None,
            'wh': [int(im.size[0]), int(im.size[1])],
            'decoded_rgb_sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
        })
    prereg = {
        'schema': 'EXPORT01-c1-prereg/1',
        'cases': cases,
        'procedure': [
            '源参考 = fp16 合并权重 + eager 子模块 + 同一 delivery_runner 编排',
            '重载 = model_dir/your_model.pt（ScriptModule）+ 同一 runner 编排',
            '每个输入各跑一次源参考与重载；512 裁切额外交错重复一次（RNG 局部协议）',
            'fp32 对照使用 R-B attempt_01 已验收的 NPY（只读，不重跑）',
        ],
        'criteria': {
            'same_precision_max_abs': 1e-5,
            'same_precision_mean_abs': 1e-6,
            'png_pixel_identical': True,
            'output_contract': 'float32/finite/[-1,1]/尺寸=输入',
            'fp16_vs_fp32_max_abs_diagnostic': 0.05,
            'fp16_vs_fp32_mean_abs_diagnostic': 0.002,
        },
        'budget': {
            'top_level_infers': 24, 'top_level_4k_infers': 8,
            'component_probes': 64, 'gpu_wall_minutes': 60,
        },
        'stop_conditions': [
            '任一输入源参考与重载超过同精度判据',
            '真实模型异常（NaN/Inf/OOM/资源缺失）',
            '预算触顶或需要改配方/精度/依赖',
        ],
    }
    out.write_text(json.dumps(prereg, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'prereg': str(out), 'sha256': file_sha(out),
                      'n_cases': len(cases)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
