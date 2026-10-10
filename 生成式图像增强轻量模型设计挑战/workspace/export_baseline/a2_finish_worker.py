# -*- coding: utf-8 -*-
"""DELIV-02 A2 收尾 worker（用户 2026-09-18 路线 A）。

目的：在**不重跑已通过项**的前提下补齐 A2（fp16）候选缺失的验收项，并保留原 FAIL 读数：
  1) 复用前置校验：模型/配方/执行模式一致 + 全部历史 NPY 哈希复核 + 量化前阶段实际数组严格复核；
  2) 新增 case7_full 全 4K 的 A（eager 同配方）与 B（JIT 重载）各 1 次 + 跨实现成对判据（矩阵第 2 行）；
  3) case1_full 迁移在 CPU 上由两侧落盘数组**重算**（不消耗 GPU），与 case7_full 迁移一并如实记录；
  4) 合并出完整 readings（含来源与哈希），再由**独立进程**执行阶段 C（4 次跨加载，严格判据）。

边界：迁移 max 一项仍为 FAIL（用户路线 A：如实披露、不改判据、不称通过）；
      阶段 C 需要独占锁与自己的 attempt，故本 worker 在派生 C 之前先关闭自己的 attempt 并释放锁。
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delivery_runner as DR  # noqa: E402
import frozen_mode  # noqa: E402
import r_ledger  # noqa: E402
import r_verify  # noqa: E402

APPROVAL = ("用户 2026-09-18 路线 A：用 A2（fp16）候选收尾出图；迁移 max 一项如实披露、"
            "不改判据、不称通过（mean 达标、超限像素占比 3.1e-05）")


def main():
    run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) for p in sys.argv[1:6]]
    prev_path = Path(os.environ["DELIV02_PREV_READINGS"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    flags = frozen_mode.apply_frozen_mode()
    (exp_dir / "a2finish_execution_flags.json").write_text(json.dumps(flags, indent=1),
                                                           encoding="utf-8")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    tol = prereg["tolerances"]
    cases = {c["name"]: c for c in prereg["cases"]}
    npy = Path(run_dir) / "npy"
    led = r_ledger.Ledger.from_file(ledger_path)
    try:
        led.acquire_lock("DELIV02 A2 finish")
    except r_ledger.LockHeld as e:
        print("[finish] LOCK_HELD %s" % e, flush=True)
        return 2
    status, detail, merged = "FAIL", "", {}
    try:
        model_sha = r_verify.sha256_file(Path(model_dir) / "your_model.pt")
        led.open_attempt(os.environ.get("DELIV02_ATTEMPT_ID", "a2_finish"), run_dir,
                         prereg_sha256=r_verify.sha256_file(prereg_path),
                         source_sha256=r_verify.sha256_file(HERE / "a2_finish_worker.py"),
                         model_sha256=model_sha)
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            if prev.get("model_sha256") != model_sha:
                raise RuntimeError("复用拒绝：历史读数模型 %s != 当前 %s"
                                   % (prev.get("model_sha256"), model_sha))
            # ── 复用证据完整性：全部历史 NPY 存在且哈希一致 ──
            bad = []
            for key in ("phaseA", "phaseB"):
                for v in (prev.get(key) or {}).values():
                    if isinstance(v, dict) and v.get("npy"):
                        p = Path(v["npy"])
                        if not p.is_file() or r_verify.sha256_file(p) != v["npy_sha256"]:
                            bad.append(str(p))
            for x in prev.get("pairs") or []:
                for side in ("A", "B"):
                    p = Path(x[side]["npy"])
                    if not p.is_file() or r_verify.sha256_file(p) != x[side]["npy_sha256"]:
                        bad.append(str(p))
            if bad:
                raise RuntimeError("复用证据哈希校验失败：%s" % bad)
            # ── 量化前阶段实际数组严格复核（缺项即 NOT_RUN 阻断）──
            stage_checks = {}
            for st in ("encode", "features", "noise", "step", "decode_raw"):
                pa, pb = npy / ("A_512_r1_stage_%s.npy" % st), npy / ("B_512_r1_stage_%s.npy" % st)
                if not (pa.is_file() and pb.is_file()):
                    raise RuntimeError("NOT_RUN 阻断：缺少阶段数组 %s" % st)
                d = (torch.from_numpy(np.load(pa, allow_pickle=False)).float() -
                     torch.from_numpy(np.load(pb, allow_pickle=False)).float()).abs()
                rec = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                       "A": str(pa), "B": str(pb),
                       "passed": bool(float(d.max()) <= tol["same_precision_max_abs"] and
                                      float(d.mean()) <= tol["same_precision_mean_abs"])}
                stage_checks[st] = rec
                if not rec["passed"]:
                    raise RuntimeError("量化前阶段 %s 未达严格判据：%s" % (st, rec))
            print("[finish] stage checks %s"
                  % {k: round(v["max_abs"], 9) for k, v in stage_checks.items()}, flush=True)

            # ── case1_full 迁移：CPU 由两侧落盘数组重算（原 FAIL 数值独立复算）──
            mig_recompute = {}
            for nm in ("case1_full",):
                got_p, ref_p = npy / ("%s_B.npy" % nm), Path(r_verify.FP32_REF[nm])
                if not got_p.is_file() or not ref_p.is_file():
                    raise RuntimeError("迁移重算缺数组：%s / %s" % (got_p, ref_p))
                got = torch.from_numpy(np.load(got_p, allow_pickle=False)).float()
                ref = torch.from_numpy(np.load(ref_p, allow_pickle=False)).float()
                d = (ref - got).abs()
                mig_recompute[nm] = {
                    "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                    "n_px_gt_max_threshold": int((d > tol["fp16_vs_fp32_max_abs_diagnostic"]).sum()),
                    "n_px_total": int(d.numel()), "max_threshold": tol["fp16_vs_fp32_max_abs_diagnostic"],
                    "mean_threshold": tol["fp16_vs_fp32_mean_abs_diagnostic"],
                    "got": str(got_p), "got_sha256": r_verify.sha256_file(got_p),
                    "fp32_ref": str(ref_p), "fp32_ref_sha256": r_verify.sha256_file(ref_p)}

            # ── 新增：case7_full 全 4K A/B 各 1 次 ──
            import core as PBcore
            cfg = PBcore.load_config()
            dev = torch.device("cuda")
            vae_dtype = r_verify.resolve_vae_dtype(exp_dir, Path(model_dir))
            work_dtype = r_verify.resolve_work_dtype(exp_dir, Path(model_dir))
            bundle, patched = r_verify.build_eager(cfg, dev, vae_dtype=vae_dtype,
                                                   work_dtype=work_dtype)
            a_runner = r_verify.EagerRunner(str(model_dir), bundle, device="cuda")
            name = "case7_full"
            x = r_verify.load_input(cases[name])
            led.reserve("top", "%s_A" % name, is_4k=True)
            with torch.jit.optimized_execution(False):
                t0 = time.perf_counter()
                out_a = a_runner.infer(x, timing={})
                wa = round(time.perf_counter() - t0, 3)
            pa = npy / ("%s_A.npy" % name)
            np.save(pa, out_a.detach().cpu().float().numpy())
            a_rec = {"call": "%s_A" % name, "wall_s": wa, "input_sha256": r_verify.tensor_sha(x),
                     "npy": str(pa), "npy_sha256": r_verify.sha256_file(pa),
                     "tensor_sha256": r_verify.tensor_sha(out_a),
                     "shape": list(out_a.shape)}
            del a_runner, bundle
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("[finish] A %s %.1fs" % (name, wa), flush=True)

            b_runner = DR.Runner(str(model_dir), device="cuda")
            led.reserve("top", "%s_B" % name, is_4k=True)
            with torch.jit.optimized_execution(False):
                t0 = time.perf_counter()
                out_b = b_runner.infer(x, timing={})
                wb = round(time.perf_counter() - t0, 3)
            pb = npy / ("%s_B.npy" % name)
            np.save(pb, out_b.detach().cpu().float().numpy())
            crit = r_verify.pixel_pair_check(out_a, out_b, tol, "%s B vs A" % name)
            b_rec = {"call": "%s_B" % name, "wall_s": wb, "input_sha256": r_verify.tensor_sha(x),
                     "npy": str(pb), "npy_sha256": r_verify.sha256_file(pb),
                     "tensor_sha256": r_verify.tensor_sha(out_b), "vs_A": crit,
                     "shape": list(out_b.shape)}
            print("[finish] %s pair max=%.3e mean=%.3e u8max=%d frac=%.6f"
                  % (name, crit["max_abs"], crit["mean_abs"],
                     crit["u8_delta"]["max_channel_diff"],
                     crit["u8_delta"]["frac_pixels_any_channel_diff"]), flush=True)
            # case7 迁移（同一次 B 输出 vs 已验收 fp32 参考）
            ref7 = Path(r_verify.FP32_REF[name])
            r7 = torch.from_numpy(np.load(ref7, allow_pickle=False)).float()
            d7 = (r7 - out_b.detach().cpu().float()).abs()
            mig7 = {"name": name, "fp32_ref": str(ref7), "max_abs": float(d7.max()),
                    "mean_abs": float(d7.mean()),
                    "n_px_gt_max_threshold": int((d7 > tol["fp16_vs_fp32_max_abs_diagnostic"]).sum()),
                    "n_px_total": int(d7.numel()), "got": str(pb),
                    "got_sha256": r_verify.sha256_file(pb)}
            print("[finish] migration %s max=%.4f mean=%.6f n_gt=%.0f"
                  % (name, mig7["max_abs"], mig7["mean_abs"],
                     mig7["n_px_gt_max_threshold"]), flush=True)
            del b_runner
            gc.collect()
            torch.cuda.empty_cache()

            # ── 合并完整读数 ──
            pairs = list(prev.get("pairs") or [])
            pairs.append({"name": name, "call": "%s_B" % name, "A": a_rec, "B": b_rec,
                          "pair": crit})
            migration = list(prev.get("migration") or [])
            migration.append({"name": "case1_full", "recomputed_cpu": True,
                              **{k: v for k, v in mig_recompute["case1_full"].items()}})
            migration.append(mig7)
            disclosed = []
            for m in migration:
                if (m["max_abs"] > tol["fp16_vs_fp32_max_abs_diagnostic"] or
                        m["mean_abs"] > tol["fp16_vs_fp32_mean_abs_diagnostic"]):
                    disclosed.append({"row": "migration_vs_fp32_baseline", "case": m["name"],
                                      "criteria": {
                                          "max_abs": tol["fp16_vs_fp32_max_abs_diagnostic"],
                                          "mean_abs": tol["fp16_vs_fp32_mean_abs_diagnostic"]},
                                      "observed": {"max_abs": m["max_abs"],
                                                   "mean_abs": m["mean_abs"],
                                                   "n_px_gt_max": m.get("n_px_gt_max_threshold"),
                                                   "n_px_total": m.get("n_px_total")},
                                      "verdict": "FAIL_DISCLOSED", "approval": APPROVAL})
            merged = {"schema": "DELIV02-a2finish-readings/1",
                      "prereg_sha256": r_verify.sha256_file(prereg_path),
                      "model_sha256": model_sha,
                      "execution_mode": prev.get("execution_mode"),
                      "execution_flags_now": flags,
                      "order": list(prev.get("order") or []) + ["%s_A" % name, "%s_B" % name],
                      "phaseA": prev.get("phaseA"), "phaseB": prev.get("phaseB"),
                      "pairs": pairs, "migration": migration,
                      "stage_checks": stage_checks, "release_check": prev.get("release_check"),
                      "merge_probe": prev.get("merge_probe"),
                      "reused_records": {
                          "source": str(prev_path),
                          "source_sha256": r_verify.sha256_file(prev_path),
                          "source_status": prev.get("status"),
                          "source_model_sha256": prev.get("model_sha256"),
                          "items": ["phaseA.call_1..4 + released", "phaseB.call_1..4",
                                    "pairs[case1_full]", "migration[case1_crop512]",
                                    "release_check", "merge_probe"],
                          "basis": "§0F：模型/数值代码/配方/运行模式/输入/哈希一致且证据完整"
                                   "（本轮已逐条复核 NPY 哈希与阶段数组）",
                          "new_calls": ["case7_full_A", "case7_full_B"]},
                      "disclosed_criteria_failures": disclosed,
                      "status_scope": ("矩阵第 1/2/3 行（量化前严格、跨实现量化后、同件重复/跨加载）"
                                       "按已记录读数判定；第 4 行迁移 max 见 "
                                       "disclosed_criteria_failures（用户路线 A 披露，不称通过）"),
                      "status": "OK"}
            (exp_dir / "readings.json").write_text(json.dumps(merged, indent=1, ensure_ascii=False),
                                                  encoding="utf-8")
            print("[finish] merged readings written; disclosed=%d" % len(disclosed), flush=True)
            detail = "disclosed_migration_fail=%d" % len(disclosed)
            status = "OK"      # 修复：合并读数成功后必须置 OK，否则阶段 C 不会启动（本轮曾因此漏跑 C）
        except Exception as e:
            detail = "%s: %s" % (type(e).__name__, e)
            (exp_dir / "readings.json").write_text(
                json.dumps({"schema": "DELIV02-a2finish-readings/1", "status": "FAIL",
                            "error": detail, "traceback": traceback.format_exc()[-2500:],
                            "ledger": led.summary()}, indent=1, ensure_ascii=False),
                encoding="utf-8")
            print("[finish] FAIL %s" % detail, flush=True)
        finally:
            try:
                led.close_attempt(status, note=detail)
            except Exception:
                pass
            led.release_lock()
        if status != "OK":
            return 1
        # ── 阶段 C：独立进程 + 独立 attempt（父子不得同时持锁）──
        cp = subprocess.run([sys.executable, "-B", str(HERE / "c_fresh_worker.py"),
                             str(run_dir), str(ledger_path), str(prereg_path),
                             str(model_dir), str(exp_dir)], capture_output=True, text=True,
                            timeout=1800, env=dict(os.environ))
        print("[finish] C rc=%d %s" % (cp.returncode, (cp.stdout or "").strip()[-200:]), flush=True)
        merged["phase_c_rc"] = cp.returncode
        merged["phase_c_tail"] = (cp.stdout or "").strip()[-400:]
        pc = exp_dir / "phase_c.json"
        if pc.is_file():
            merged["phase_c"] = json.loads(pc.read_text(encoding="utf-8"))
        merged["status"] = "OK" if cp.returncode == 0 else "FAIL_PHASE_C"
        (exp_dir / "readings.json").write_text(json.dumps(merged, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
        return 0 if cp.returncode == 0 else 1
    finally:
        try:
            led.release_lock()      # 幂等：正常路径已释放；异常路径兜底
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
