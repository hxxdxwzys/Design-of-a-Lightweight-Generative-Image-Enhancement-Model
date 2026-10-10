# -*- coding: utf-8 -*-
"""DELIV-02 FP32 回退路由的 CPU 冒烟（不碰 GPU、不碰总账）。

覆盖本次新增/修改的实际路径：
  - deliv_worker.resolve_model_dir / gate_call_count（FP32 候选目录与精确预留）
  - r3_component_gate._repeat_selection（重复项选择，默认全量）
  - r_verify.resolve_vae_dtype / resolve_work_dtype（全 FP32 / FP32-VAE / 缺项拒绝）
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EB = HERE.parent
if str(EB) not in sys.path:
    sys.path.insert(0, str(EB))

dw = importlib.import_module("deliv_worker")
cg = importlib.import_module("r3_component_gate")
rv = importlib.import_module("r_verify")


def test_resolve_model_dir_keeps_a2_dir():
    run = Path("C:/models/exports/DELIV02_X")
    a2 = Path("C:/models/exports/DELIV02_X/model_dir")
    assert dw.resolve_model_dir(run, a2, False) == a2
    fp = dw.resolve_model_dir(run, a2, True)
    assert fp == run / "model_dir_fp32"
    assert fp != a2                      # 绝不落在已冻结的 A2 目录


def test_gate_call_count_matches_real_forwards():
    # 检查侧真实前向：6 模块 + 3 step + 2x 模块重复 + 3x step 重复
    assert dw.gate_call_count(None) == 24
    assert dw.gate_call_count({"enc_512", "dec_64"}) == 9 + 4
    assert dw.gate_call_count({"step"}) == 12
    assert dw.gate_call_count(set()) == 9


def test_repeat_selection_default_is_all(monkeypatch):
    monkeypatch.delenv("DELIV_GATE_REPEATS", raising=False)
    assert cg._repeat_selection() is None
    monkeypatch.setenv("DELIV_GATE_REPEATS", " enc_512 , dec_64 ")
    assert cg._repeat_selection() == {"enc_512", "dec_64"}
    monkeypatch.setenv("DELIV_GATE_REPEATS", "   ")
    assert cg._repeat_selection() is None


def _mk(tmp_path, md):
    d = tmp_path / "model_dir_fp32"
    d.mkdir()
    (d / "models_export.json").write_text(
        json.dumps({"module_dtypes": md, "weight_dtype": "float32"}), encoding="utf-8")
    return d, tmp_path / "exp"


def test_dtype_resolution_full_fp32(tmp_path):
    d, exp = _mk(tmp_path, {"vae_encoder": "torch.float32", "dit": "torch.float32",
                            "dino": "torch.float32", "light_decoder": "torch.float32"})
    import torch
    assert rv.resolve_vae_dtype(exp, d) is torch.float32
    assert rv.resolve_work_dtype(exp, d) is torch.float32


def test_dtype_resolution_a2_recipe(tmp_path):
    d, exp = _mk(tmp_path, {"vae_encoder": "torch.float32", "dit": "torch.float16",
                            "dino": "torch.float16", "light_decoder": "torch.float32"})
    import torch
    assert rv.resolve_vae_dtype(exp, d) is torch.float32
    assert rv.resolve_work_dtype(exp, d) is torch.float16


def test_dtype_resolution_refuses_missing(tmp_path):
    d = tmp_path / "model_dir_fp32"
    d.mkdir()
    (d / "models_export.json").write_text(json.dumps({"weight_dtype": "float32"}),
                                         encoding="utf-8")
    with pytest.raises(RuntimeError):
        rv.resolve_vae_dtype(tmp_path / "exp", d)
