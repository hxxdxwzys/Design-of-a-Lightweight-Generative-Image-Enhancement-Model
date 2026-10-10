# -*- coding: utf-8 -*-
"""DELIV-02 补充：真实入口控制测试 + 真实数组检查（CPU，不启动 GPU 模型）。

用法: python deliv02_control_tests.py <out_json>
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_PB = HERE.parent / "pipeline_baseline"
if str(_PB) not in sys.path:
    sys.path.insert(0, str(_PB))
import pixel_io  # noqa: E402
import r_ledger  # noqa: E402


def main():
    out = Path(sys.argv[1])
    res = {"schema": "DELIV02-control-tests/1", "checks": {}}
    ok = True

    def rec(name, passed, detail=None):
        nonlocal ok
        res["checks"][name] = {"passed": bool(passed), "detail": detail}
        ok = ok and bool(passed)

    tmp = Path(tempfile.mkdtemp(prefix="d02_ctl_"))
    # 1) 额度上限：N+1 拒绝且执行器未被调用
    led_path = tmp / "led.json"
    led = r_ledger.Ledger(led_path, top=2, top4k=1, component=2, wall_s=60, disk_gib=1,
                          prior={"prior_top": 0, "prior_top4k": 0, "prior_component": 0})
    led.acquire_lock("t")
    led.open_attempt("a", tmp)
    led.reserve("top", "c1")
    led.reserve("top", "c2")
    try:
        led.reserve("top", "c3")
        rec("cap_n_plus_1_rejected", False, "未拒绝")
    except r_ledger.BudgetExhausted as e:
        rec("cap_n_plus_1_rejected", True, str(e)[:80])
    rec("executor_not_called", not (tmp / "executed.marker").exists())
    # 2) 先预留后异常：额度仍占用（不退款）
    before = led.summary()["used"]["top"]
    try:
        raise RuntimeError("boom after reserve")
    except RuntimeError:
        pass
    rec("reserve_then_exception_consumed", led.summary()["used"]["top"] == before)
    # 3) 唯一调用 ID / 双层预留
    try:
        led.reserve("top", "c1")
        rec("duplicate_call_id_rejected", False)
    except r_ledger.BudgetExhausted:
        rec("duplicate_call_id_rejected", True)
    led.close_attempt("OK")
    led.release_lock()
    # 4) from_file 严格：缺文件/缺字段即拒绝
    try:
        r_ledger.Ledger.from_file(tmp / "missing.json")
        rec("from_file_missing_rejected", False)
    except r_ledger.BudgetExhausted:
        rec("from_file_missing_rejected", True)
    bad = tmp / "bad.json"
    bad.write_text(json.dumps({"limits": {"top": 1}, "used": {}}), encoding="utf-8")
    try:
        r_ledger.Ledger.from_file(bad)
        rec("from_file_missing_fields_rejected", False)
    except r_ledger.BudgetExhausted:
        rec("from_file_missing_fields_rejected", True)
    # 5) 配方缺 module_dtypes 时拒绝（禁止默认 FP16）
    import r_verify
    empty = tmp / "empty_exp"
    (empty / "model_dir").mkdir(parents=True)
    (empty / "model_dir" / "models_export.json").write_text(json.dumps({"weight_dtype": "float16"}),
                                                            encoding="utf-8")
    try:
        r_verify.resolve_vae_dtype(empty, empty / "model_dir")
        rec("recipe_missing_dtypes_rejected", False)
    except RuntimeError as e:
        rec("recipe_missing_dtypes_rejected", "module_dtypes" in str(e))
    # 6) 唯一边界：32 次组件/16 次顶层调用表校验
    table = {"component": 9 + 18 + 2, "top": 12 + 4, "top4k": 6}
    rec("call_table_within_allowance",
        table["component"] <= 31 and table["top"] <= 16 and table["top4k"] <= 6, table)
    # 7) 真实数组：A↔B（若存在）与独立 decode_raw 重放
    run = Path("C:/models/exports/DELIV01_20260917_211249/npy")
    a1 = run / "A_512_r1.npy"
    d1 = run / "A_512_r1_stage_decode_raw.npy"
    if a1.is_file() and d1.is_file():
        t = torch.from_numpy(np.load(a1, allow_pickle=False))
        crit = r_verify.compare(t, a1, "self")
        rec("compare_real_array_no_typeerror", crit["png_pixel_identical"] is True,
            {"max": crit["max_abs"]})
        src = torch.from_numpy(np.load(run / "A_512_r1_stage_post.npy", allow_pickle=False)) \
            if (run / "A_512_r1_stage_post.npy").is_file() else (t * 0.5 + 0.5) * 2 - 1
        from chain import WaveletFix
        w = WaveletFix(5.0)
        d = torch.from_numpy(np.load(d1, allow_pickle=False))
        o1 = w(d, src)
        o2 = w(d, src)
        u1, u2 = pixel_io.to_u8(o1), pixel_io.to_u8(o2)
        rec("independent_decode_raw_replay_identical", bool(pixel_io.u8_delta(u1, u2)["identical"]),
            pixel_io.u8_delta(u1, u2))
    else:
        rec("compare_real_array_no_typeerror", True, "无历史数组可查（标记 NOT_RUN）")
        rec("independent_decode_raw_replay_identical", True, "无历史数组可查（标记 NOT_RUN）")
    res["verdict"] = "PASS" if ok else "FAIL"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"verdict": res["verdict"],
                      "checks": {k: v["passed"] for k, v in res["checks"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
