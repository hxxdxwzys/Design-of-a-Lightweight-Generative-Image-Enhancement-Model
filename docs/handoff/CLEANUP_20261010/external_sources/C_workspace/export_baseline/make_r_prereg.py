# -*- coding: utf-8 -*-
"""EXPORT-01R 预登记（先登记后执行）：固定输入、调用计划、判据、预算、执行模式、停止条件。

用法: python make_r_prereg.py <out_json>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TEST_DIR = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1])
    cases = []
    for name, src, box in (
            ("case1_crop512", "case1.jpg", (1792, 1280, 2304, 1792)),
            ("case1_crop511x513", "case1.jpg", (1792, 1280, 1792 + 513, 1280 + 511)),
            ("case1_full", "case1.jpg", None),
            ("case7_full", "case7.jpg", None)):
        p = TEST_DIR / src
        im = Image.open(p).convert("RGB")
        if box:
            im = im.crop(box)
        arr = np.asarray(im)
        cases.append({"name": name, "source_file": str(p), "source_sha256": file_sha(p),
                      "crop_box": list(box) if box else None,
                      "wh": [int(im.size[0]), int(im.size[1])],
                      "decoded_rgb_sha256": hashlib.sha256(arr.tobytes()).hexdigest()})
    prereg = {
        "schema": "EXPORT01R-prereg/1",
        "cases": cases,
        "call_plan": {
            "A_512_repeat": 3, "A_511x513": 1, "B_512_repeat": 3, "B_511x513": 1,
            "pairs_4k": ["case1_full", "case7_full"],
            "order": "A(512x3,511x513) -> release A -> B(512x3,511x513) -> case1 4K A/B -> case7 4K A/B",
            "top_limit": 12, "top4k_limit": 4},
        "batch": (sys.argv[2] if len(sys.argv) > 2 else "EXPORT01R2"),
        "merge_probe_criteria": {"max_abs": 1e-5, "mean_abs": 1e-6, "finite": True,
                                 "note": "fp32 组件同输入对照（peft 未合并 vs merged），无 PNG 要求"},
        "tolerances": {"same_precision_max_abs": 1e-5, "same_precision_mean_abs": 1e-6,
                       "fp16_vs_fp32_max_abs_diagnostic": 0.05,
                       "fp16_vs_fp32_mean_abs_diagnostic": 0.002,
                       "wavelet_cpu_max_abs_vs_reference": 0.007843137254901961,
                       "wavelet_cpu_frac_pixels_diff_max": 0.001},
        "execution_mode": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                           "torch.use_deterministic_algorithms": "True(warn_only=False)",
                           "cudnn_benchmark": False, "cudnn_deterministic": True,
                           "cudnn_allow_tf32": False, "matmul_allow_tf32": False,
                           "torch.jit.optimized_execution": False},
        "required_evidence": ["双方与重复原始NPY及sha256", "每阶段输出的sha256/finite",
                              "监督记录(pid/退出码/超时)", "不可退款事件总账",
                              "bundle_manifest与bundle_id", "执行模式开关记录"],
        "stop_conditions": ["首个真实异常", "输出/重复/成对/迁移判据失败", "身份或源码漂移",
                            "预算触顶且下一步需新额度", "磁盘水位不足", "监督失败"],
        "budget": {"top": 12, "top4k": 4, "component": 48, "wall_s": 1800, "disk_gib": 8},
    }
    out.write_text(json.dumps(prereg, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"prereg": str(out), "sha256": file_sha(out), "n_cases": len(cases)}))


if __name__ == "__main__":
    main()
