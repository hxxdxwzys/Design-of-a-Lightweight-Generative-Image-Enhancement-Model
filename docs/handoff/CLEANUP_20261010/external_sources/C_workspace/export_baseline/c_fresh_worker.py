# -*- coding: utf-8 -*-
"""DELIV-02 阶段 C：全新进程加载同一 .pt，跑 4 次并与 B 比对跨加载保真（严格判据）。"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delivery_runner as DR  # noqa: E402
import frozen_mode  # noqa: E402
import pixel_io  # noqa: E402
import r_ledger  # noqa: E402
import r_verify  # noqa: E402


def main():
    run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) for p in sys.argv[1:6]]
    frozen_mode.apply_frozen_mode()
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    cases = {c["name"]: c for c in prereg["cases"]}
    led = r_ledger.Ledger.from_file(ledger_path)   # DELIV-02：额度以磁盘总账为准
    try:
        led.acquire_lock("DELIV02 phase C")
    except r_ledger.LockHeld as e:
        print("[C] LOCK_HELD %s" % e, flush=True)
        return 2
    status, detail = "FAIL", ""
    npy_dir = Path(run_dir) / "npy"
    try:
        _att = os.environ.get("DELIV02_ATTEMPT_ID", "attempt_c")
        led.open_attempt(_att, run_dir,
                         prereg_sha256=r_verify.sha256_file(prereg_path),
                         source_sha256=r_verify.sha256_file(HERE / "c_fresh_worker.py"),
                         model_sha256=r_verify.sha256_file(Path(model_dir) / "your_model.pt"))
        try:
            # DELIV-02 补充 1/53：C 只能在前置条件满足后加载——A/B 完整 PASS、四份 B 参考存在且哈希正确、绑定同一 revision/bundle；
            # 失败时 C 执行器调用次数必须为 0（旧 C 产物/失败不得覆盖）。
            rd_p = exp_dir / "readings.json"
            if not rd_p.is_file():
                raise RuntimeError("阶段门禁：缺少 A/B readings.json，C 不得启动")
            rd = json.loads(rd_p.read_text(encoding="utf-8"))
            if rd.get("status") != "OK":
                raise RuntimeError("阶段门禁：A/B 未 PASS（status=%s），C 执行器调用 0 次"
                                   % rd.get("status"))
            if rd.get("model_sha256") != r_verify.sha256_file(Path(model_dir) / "your_model.pt"):
                raise RuntimeError("阶段门禁：A/B 绑定的模型与本 revision 不一致")
            need_b = ["B_512_r1", "B_511x513", "case1_full_B", "case7_full_B"]
            bad = []
            for nm in need_b:
                ref = None
                for key in ("phaseB", "pairs"):
                    node = rd.get(key) or {}
                    for v in (node.values() if isinstance(node, dict) else node):
                        if isinstance(v, dict) and v.get("call") == nm.replace("B_512_r1", "B_512_1") \
                                and v.get("npy"):
                            ref = v
                p = npy_dir / ("%s.npy" % nm)
                if not p.is_file():
                    bad.append("%s 缺失" % nm)
                    continue
                if ref is not None and ref.get("npy_sha256") and \
                        r_verify.sha256_file(p) != ref["npy_sha256"]:
                    bad.append("%s 哈希不符" % nm)
            if bad:
                raise RuntimeError("阶段门禁：B 参考校验失败 %s，C 执行器调用 0 次" % bad)
            runner = DR.Runner(str(model_dir), device="cuda")
            out = {}
            plan = [("C_512", "case1_crop512", False, "B_512_r1"),
                    ("C_511x513", "case1_crop511x513", False, "B_511x513"),
                    ("C_case1_full", "case1_full", True, "case1_full_B"),
                    ("C_case7_full", "case7_full", True, "case7_full_B")]
            checks = {}
            for call, case, is4k, b_name in plan:
                led.reserve("top", call, is_4k=is4k)
                x = r_verify.load_input(cases[case])
                with torch.jit.optimized_execution(False):
                    t0 = __import__("time").perf_counter()
                    y = runner.infer(x, timing={})
                    wall = round(__import__("time").perf_counter() - t0, 3)
                p = npy_dir / ("%s.npy" % call)
                np.save(p, y.detach().cpu().float().numpy())
                ref = torch.from_numpy(np.load(npy_dir / ("%s.npy" % b_name),
                                               allow_pickle=False)).to(y.device)
                d = (ref.float() - y.float()).abs()
                a_u8, b_u8 = pixel_io.to_u8(ref), pixel_io.to_u8(y)
                delta = pixel_io.u8_delta(a_u8, b_u8)
                ok = (float(d.max()) <= 1e-5 and float(d.mean()) <= 1e-6 and
                      delta["identical"])
                checks[call] = {"vs_B": b_name, "max_abs": float(d.max()),
                                "mean_abs": float(d.mean()), "u8": delta,
                                "wall_s": wall, "passed": bool(ok),
                                "npy": str(p), "npy_sha256": r_verify.sha256_file(p)}
                print("[C] %s max=%.3e mean=%.3e u8_identical=%s"
                      % (call, checks[call]["max_abs"], checks[call]["mean_abs"],
                         delta["identical"]), flush=True)
                if not ok:
                    raise RuntimeError("%s 跨加载未达严格判据" % call)
            (exp_dir / "phase_c.json").write_text(
                json.dumps({"schema": "DELIV02-phase-c/1", "checks": checks,
                            "verdict": "PASS"}, indent=1, ensure_ascii=False), encoding="utf-8")
            status = "OK"
        except Exception as e:
            detail = "%s: %s" % (type(e).__name__, e)
            (exp_dir / "phase_c.json").write_text(
                json.dumps({"schema": "DELIV02-phase-c/1", "error": detail,
                            "ledger": led.summary(), "verdict": "FAIL"}, indent=1,
                           ensure_ascii=False), encoding="utf-8")
            print("[C] FAIL %s" % detail, flush=True)
    finally:
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
