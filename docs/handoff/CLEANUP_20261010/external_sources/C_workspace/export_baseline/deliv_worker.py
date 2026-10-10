# -*- coding: utf-8 -*-
"""DELIV-01 P1 worker：组件门槛（匹配条件）→ 12 次完整成对验证。"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # 必须在 CUDA 初始化前

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import frozen_mode  # noqa: E402
import r_ledger  # noqa: E402
import r_verify  # noqa: E402


def resolve_model_dir(run_dir, model_dir, full_fp32):
    """FP32 回退候选固定落在 <run_dir>/model_dir_fp32，绝不覆盖已冻结的 A2 model_dir。"""
    return (Path(run_dir) / "model_dir_fp32") if full_fp32 else Path(model_dir)


def gate_call_count(selection=None):
    """组件门槛检查侧真实前向数：6 模块 + 3 step + 2×模块重复 + 3×(step 重复)。

    旧实现按固定 18 预留而实际需要 24（少记 6），此处按实际口径精确预留。
    """
    mods = ('enc_512', 'enc_1024', 'dec_64', 'dec_128', 'dino', 'wavelet')
    n_mod = len([k for k in mods if selection is None or k in selection])
    n_step = 3 if (selection is None or 'step' in selection) else 0
    return 9 + 2 * n_mod + n_step


def main():
    run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) for p in sys.argv[1:6]]
    exp_dir.mkdir(parents=True, exist_ok=True)
    flags = frozen_mode.apply_frozen_mode()
    (exp_dir / "p1_execution_flags.json").write_text(json.dumps(flags, indent=1), encoding="utf-8")
    led = r_ledger.Ledger.from_file(ledger_path)   # DELIV-02：额度以磁盘总账为准
    try:
        led.acquire_lock("DELIV01 P1")
    except r_ledger.LockHeld as e:
        print("[worker] LOCK_HELD %s" % e, flush=True)
        return 2
    status, detail = "FAIL", ""
    # DELIV-02 2026-09-18：全 FP32 数值回退候选（§0F 第二段预授权，唯一第二候选）
    full_fp32 = os.environ.get("DELIV02_FULL_FP32", "") == "1"
    target_dir = resolve_model_dir(run_dir, model_dir, full_fp32)
    os.environ["DELIV_MODEL_SUBDIR"] = target_dir.name
    try:
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        model_p = target_dir / "your_model.pt"
        _att = os.environ.get("DELIV02_ATTEMPT_ID", "attempt_01")
        led.open_attempt(_att, run_dir,
                         prereg_sha256=r_verify.sha256_file(prereg_path),
                         source_sha256=r_verify.sha256_file(HERE / "r_verify.py"),
                         model_sha256=(r_verify.sha256_file(model_p) if model_p.is_file()
                                       else None))
        try:
            import gc
            built_here = False
            build_info = None
            if not model_p.is_file():
                if not full_fp32:
                    raise RuntimeError("DELIV-02 路径禁止自动构建：模型缺失即拒绝")
                import r_build
                led.note("fp32_fallback_build_start", {"model_dir": str(target_dir),
                                                       "run_dir": str(run_dir)})
                build_info = r_build.build(target_dir, run_dir, led, exp_dir, None, HERE,
                                           full_fp32=True)
                built_here = True
                led.note("fp32_fallback_build_done",
                         {k: build_info.get(k) for k in
                          ("model_sha256", "model_bytes", "bundle_id", "module_dtypes")})
                print("[worker] FP32 build sha=%s bytes=%s bundle=%s"
                      % (build_info["model_sha256"], build_info["model_bytes"],
                         build_info["bundle_id"]), flush=True)
            else:
                led.note("fp32_fallback_reuse",
                         {"model_sha256": r_verify.sha256_file(model_p)})
            model_dir = target_dir                    # 后续所有路径都以本候选为准
            # ── 精度来自配方（缺项即拒绝，禁止默认 FP16）──
            recipe = json.loads((target_dir / "models_export.json").read_text(encoding="utf-8"))
            _md = recipe.get("module_dtypes") or {}
            if "vae_encoder" not in _md or "dit" not in _md:
                raise RuntimeError("INVALID_PROTOCOL: 配方缺少 module_dtypes，禁止默认 FP16")
            vae_dtype = (torch.float32 if str(_md["vae_encoder"]).endswith("float32")
                         else torch.float16)
            work_dtype = (torch.float32 if str(_md["dit"]).endswith("float32")
                          else torch.float16)
            # ── 组件门槛：eager 参考（与导出件同配方；build 已落盘时直接复用）──
            if built_here:
                meta = build_info["component_refs"]
            else:
                import core as PBcore
                led.reserve("component", "p1_load_assets_for_reference")
                cfg = PBcore.load_config()
                assets = PBcore.Assets(cfg, torch.device("cuda"))
                merged = assets.model.merge_and_unload().to(work_dtype)
                from chain import (Decode, DinoFeat, DitStep, VAEEncode, WaveletFix,
                                   patch_device_safe_timestep, patch_fp16_embedders)
                patch_fp16_embedders(merged, work_dtype)
                patch_device_safe_timestep(merged)
                dev = torch.device("cuda")
                wraps = (VAEEncode(assets.vae.to(vae_dtype), vae_dtype,
                                   work_dtype).to(dev).eval(),
                         Decode(assets.dec.to(vae_dtype), assets.vae.config.scaling_factor,
                                vae_dtype, torch.float32).to(dev).eval(),
                         DinoFeat(assets.venc.to(work_dtype),
                                  size=int(cfg["expected_dinov2_size"]),
                                  layer_index=int(cfg["expected_layer_dinov2b_list"][0]),
                                  n_blocks=len(assets.venc.blocks)).to(dev).eval(),
                         DitStep(merged).to(dev).eval(),
                         WaveletFix(sigma=float(cfg["align_sigma"])).to(dev).eval())
                import r3_component_gate
                meta = r3_component_gate.capture_references(wraps, dev,
                                                            exp_dir / "component_refs", led,
                                                            work_dtype)
                del wraps, merged, assets
                gc.collect()
                torch.cuda.empty_cache()
            print("[worker] references captured: %d" % len(meta["refs"]), flush=True)
            # ── 全新进程组件检查（真实前向，先扣后跑；预留数按实际口径）──
            import r3_component_gate as _cg
            sel = _cg._repeat_selection()
            n_gate = gate_call_count(sel)
            for i in range(n_gate):
                led.reserve("component", "fp32_gate_component_%d" % (i + 1))
            led.note("component_gate_plan", {"model_dir": str(target_dir),
                                             "built_here": built_here,
                                             "n_reserved": n_gate,
                                             "repeats": (sorted(sel) if sel else "ALL"),
                                             "work_dtype": str(work_dtype),
                                             "vae_dtype": str(vae_dtype)})
            cp = subprocess.run([sys.executable, "-B", str(HERE / "r3_component_gate.py"),
                                 "check", str(run_dir), str(exp_dir), str(prereg_path)],
                                capture_output=True, text=True, timeout=1800,
                                env=dict(os.environ))
            led.note("component_gate", {"rc": cp.returncode, "tail": (cp.stdout or "")[-300:]})
            print("[worker] component_gate rc=%d %s"
                  % (cp.returncode, (cp.stdout or "").strip()[-200:]), flush=True)
            if cp.returncode != 0:
                raise RuntimeError("组件门槛未通过：%s" % ((cp.stderr or cp.stdout or "")[-400:]))
            led.note("component_gate_pass", meta["ref_sha256"])
            # ── 12 次完整成对验证 ──
            led.reserve("component", "p1_verification_asset_loads")
            readings = r_verify.run_verification(run_dir, led, prereg_path, model_dir,
                                                 prereg["tolerances"], exp_dir)
            readings["status"] = "OK"
            (exp_dir / "readings.json").write_text(
                json.dumps(readings, indent=1, ensure_ascii=False), encoding="utf-8")
            status = "OK"
        except Exception as e:
            detail = "%s: %s" % (type(e).__name__, e)
            partial = {}
            rp = exp_dir / "readings.json"
            if rp.is_file():
                try:
                    partial = json.loads(rp.read_text(encoding="utf-8"))
                except Exception:
                    partial = {}
            partial.update({"schema": "DELIV01-p1-readings/1", "status": "FAIL", "error": detail,
                            "traceback": traceback.format_exc()[-4000:],
                            "ledger": led.summary(), "elapsed_s": round(led.elapsed_s(), 3)})
            rp.write_text(json.dumps(partial, indent=1, ensure_ascii=False), encoding="utf-8")
            print("[worker] P1 FAIL %s" % detail, flush=True)
    finally:
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
