# -*- coding: utf-8 -*-
"""DELIV-02 续跑 worker：在已完成的 A 侧与 B_512_1 基础上，补齐 B 其余调用、4K A/B，并触发全新进程 C。

剩余额度精确对应：A 4K×2 + B(512×2 + odd + 4K×2) + C(4) = 11 次顶层（4K 6）。
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
import pixel_io  # noqa: E402
import r_ledger  # noqa: E402
import r_verify  # noqa: E402


def main():
    run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) for p in sys.argv[1:6]]
    frozen_mode.apply_frozen_mode()
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    tol = prereg["tolerances"]
    cases = {c["name"]: c for c in prereg["cases"]}
    npy = Path(run_dir) / "npy"
    led = r_ledger.Ledger.from_file(ledger_path)
    try:
        led.acquire_lock("DELIV02 resume")
    except r_ledger.LockHeld as e:
        print("[resume] LOCK_HELD %s" % e, flush=True)
        return 2
    status, detail = "FAIL", ""
    try:
        led.open_attempt(os.environ.get("DELIV02_ATTEMPT_ID", "cont2_resume"), run_dir,
                         prereg_sha256=r_verify.sha256_file(prereg_path),
                         source_sha256=r_verify.sha256_file(HERE / "d02_resume_worker.py"),
                         model_sha256=r_verify.sha256_file(Path(model_dir) / "your_model.pt"))
        try:
            rd = json.loads((exp_dir / "readings.json").read_text(encoding="utf-8"))
            # 前置：A 四次与 B_512_1 的跨实现判定已通过，且 NPY 哈希一致
            a1 = rd["phaseA"]["call_1"]
            b1 = rd["phaseB"]["call_1"]
            assert b1["call"] == "B_512_1" and b1["vs_A"]["passed"], "B_512_1 跨实现未通过"
            for k, v in rd["phaseA"].items():
                if isinstance(v, dict) and v.get("npy"):
                    p = Path(v["npy"])
                    assert p.is_file() and r_verify.sha256_file(p) == v["npy_sha256"], k
            # §0F.4：量化前关键阶段的**实际数组**严格对照（流式、同阶段）
            stage_checks = {}
            # 严格浮点判据仅适用于**量化前**阶段；post 是 wavelet 量化后的输出，',
            # 归 DELIV-02 矩阵第 2 行（允许量化边界差异，max≤2/255+1e-6）',
            for st in ("encode", "features", "noise", "step", "decode_raw"):
                pa = npy / ("A_512_r1_stage_%s.npy" % st)
                pb = npy / ("B_512_r1_stage_%s.npy" % st)
                if pa.is_file() and pb.is_file():
                    ta = torch.from_numpy(np.load(pa, allow_pickle=False))
                    tb = torch.from_numpy(np.load(pb, allow_pickle=False))
                    d = (ta.float() - tb.float()).abs()
                    ok = float(d.max()) <= 1e-5 and float(d.mean()) <= 1e-6
                    stage_checks[st] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                        "passed": bool(ok)}
                    if not ok:
                        raise RuntimeError("量化前阶段 %s 未达 max1e-5/mean1e-6：%s"
                                           % (st, stage_checks[st]))
            print("[resume] stage checks %s" % {k: round(v["max_abs"], 9)
                                                for k, v in stage_checks.items()}, flush=True)

            import core as PBcore
            from chain import Decode, DinoFeat, DitStep, VAEEncode, WaveletFix
            from chain import patch_device_safe_timestep, patch_fp16_embedders
            cfg = PBcore.load_config()
            dev = torch.device("cuda")
            assets = PBcore.Assets(cfg, dev)
            vae_dtype = r_verify.resolve_vae_dtype(exp_dir, Path(model_dir))
            merged = assets.model.merge_and_unload().to(torch.float16)
            patch_fp16_embedders(merged, torch.float16)
            patch_device_safe_timestep(merged)
            a_bundle = r_verify.EagerBundle(
                VAEEncode(assets.vae.to(vae_dtype), vae_dtype, torch.float16).to(dev).eval(),
                Decode(assets.dec.to(vae_dtype), assets.vae.config.scaling_factor, vae_dtype,
                       torch.float32).to(dev).eval(),
                DinoFeat(assets.venc.to(torch.float16),
                         size=int(cfg["expected_dinov2_size"]),
                         layer_index=int(cfg["expected_layer_dinov2b_list"][0]),
                         n_blocks=len(assets.venc.blocks)).to(dev).eval(),
                DitStep(merged).to(dev).eval(),
                WaveletFix(sigma=float(cfg["align_sigma"])).to(dev).eval())
            a_runner = r_verify.EagerRunner(str(model_dir), a_bundle, device="cuda")
            res = {"schema": "DELIV02-resume-readings/1", "stage_checks": stage_checks,
                   "pairs": [], "order": []}

            def pair(name, ref_npy, out):
                ref = torch.from_numpy(np.load(ref_npy, allow_pickle=False)).to(out.device)
                return r_verify.pixel_pair_check(ref, out, tol, name)

            # A：两张 4K（各 1 次）
            for name in ("case1_full", "case7_full"):
                x = r_verify.load_input(cases[name])
                led.reserve("top", "%s_A" % name, is_4k=True)
                with torch.jit.optimized_execution(False):
                    t0 = time.perf_counter()
                    out_a = a_runner.infer(x, timing={})
                    wa = round(time.perf_counter() - t0, 3)
                p = npy / ("%s_A.npy" % name)
                np.save(p, out_a.detach().cpu().float().numpy())
                res["order"].append({"call": "%s_A" % name, "wall_s": wa,
                                     "npy_sha256": r_verify.sha256_file(p)})
                print("[resume] A %s %.1fs" % (name, wa), flush=True)
            del a_runner, a_bundle
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # B：512 第 2/3 次、奇数裁切、两张 4K（共 5 次）
            b_runner = DR.Runner(str(model_dir), device="cuda")
            for i in (2, 3):
                x = r_verify.load_input(cases["case1_crop512"])
                led.reserve("top", "B_512_%d" % i)
                with torch.jit.optimized_execution(False):
                    out = b_runner.infer(x, timing={})
                p = npy / ("B_512_r%d.npy" % i)
                np.save(p, out.detach().cpu().float().numpy())
                crit = pair("B_512_%d vs A" % i, npy / ("A_512_r%d.npy" % i), out)
                rep = r_verify.compare(out, npy / "B_512_r1.npy", "B_512_%d vs B_512_r1" % i)
                r_verify.check(rep, tol, "B_512_%d 自重复" % i)
                res["pairs"].append({"name": "case1_crop512_r%d" % i, "pair": crit,
                                     "self_repeat": rep})
                print("[resume] B_512_%d max=%.3e mean=%.3e u8max=%d frac=%.6f"
                      % (i, crit["max_abs"], crit["mean_abs"],
                         crit["u8_delta"]["max_channel_diff"],
                         crit["u8_delta"]["frac_pixels_any_channel_diff"]), flush=True)
            x = r_verify.load_input(cases["case1_crop511x513"])
            led.reserve("top", "B_511x513")
            with torch.jit.optimized_execution(False):
                out = b_runner.infer(x, timing={})
            p = npy / "B_511x513.npy"
            np.save(p, out.detach().cpu().float().numpy())
            crit = pair("B_511x513 vs A", npy / "A_511x513.npy", out)
            res["pairs"].append({"name": "case1_crop511x513", "pair": crit})
            for name in ("case1_full", "case7_full"):
                x = r_verify.load_input(cases[name])
                led.reserve("top", "%s_B" % name, is_4k=True)
                with torch.jit.optimized_execution(False):
                    t0 = time.perf_counter()
                    out_b = b_runner.infer(x, timing={})
                    wb = round(time.perf_counter() - t0, 3)
                p = npy / ("%s_B.npy" % name)
                np.save(p, out_b.detach().cpu().float().numpy())
                crit = pair("%s B vs A" % name, npy / ("%s_A.npy" % name), out_b)
                res["pairs"].append({"name": name, "pair": crit, "wall_s": wb,
                                     "npy_sha256": r_verify.sha256_file(p)})
                print("[resume] %s max=%.3e mean=%.3e u8max=%d frac=%.6f"
                      % (name, crit["max_abs"], crit["mean_abs"],
                         crit["u8_delta"]["max_channel_diff"],
                         crit["u8_delta"]["frac_pixels_any_channel_diff"]), flush=True)
            del b_runner
            gc.collect()
            torch.cuda.empty_cache()

            # C：全新进程（其自身有 A/B PASS 前置检查）
            cp = subprocess.run([sys.executable, "-B", str(HERE / "c_fresh_worker.py"),
                                 str(run_dir), str(ledger_path), str(prereg_path),
                                 str(model_dir), str(exp_dir)], capture_output=True, text=True,
                                timeout=900)
            res["phase_c_rc"] = cp.returncode
            res["phase_c_tail"] = (cp.stdout or "")[-300:]
            print("[resume] C rc=%d" % cp.returncode, flush=True)
            if cp.returncode != 0:
                raise RuntimeError("C 阶段失败：%s" % ((cp.stderr or cp.stdout)[-300:]))
            res["status"] = "OK"
            (exp_dir / "readings_resume.json").write_text(
                json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
            status = "OK"
        except Exception as e:
            detail = "%s: %s" % (type(e).__name__, e)
            (exp_dir / "readings_resume.json").write_text(
                json.dumps({"status": "FAIL", "error": detail,
                            "traceback": traceback.format_exc()[-2500:],
                            "ledger": led.summary()}, indent=1, ensure_ascii=False),
                encoding="utf-8")
            print("[resume] FAIL %s" % detail, flush=True)
    finally:
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
