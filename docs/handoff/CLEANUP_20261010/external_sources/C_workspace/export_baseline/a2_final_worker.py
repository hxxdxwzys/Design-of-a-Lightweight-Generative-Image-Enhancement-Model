# -*- coding: utf-8 -*-
"""DELIV-01R A2 收口 worker：同配方组件对照（enc/dec）→ 12 次完整成对验证。"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_mode  # noqa: E402
import r_ledger  # noqa: E402
import r_verify  # noqa: E402


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) for p in sys.argv[1:6]]
    exp_dir.mkdir(parents=True, exist_ok=True)
    flags = frozen_mode.apply_frozen_mode()
    (exp_dir / "a2_execution_flags.json").write_text(json.dumps(flags, indent=1),
                                                    encoding="utf-8")
    led = r_ledger.Ledger(ledger_path)
    try:
        led.acquire_lock("DELIV01R A2 final")
    except r_ledger.LockHeld as e:
        print("[a2] LOCK_HELD %s" % e, flush=True)
        return 2
    status, detail = "FAIL", ""
    try:
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        tol = prereg["tolerances"]
        mp = Path(model_dir) / "your_model.pt"
        led.open_attempt("attempt_01", run_dir, prereg_sha256=sha(prereg_path),
                         source_sha256=sha(HERE / "a2_final_worker.py"),
                         model_sha256=sha(mp))
        try:
            import core as PBcore
            from chain import Decode, VAEEncode
            recipe = json.loads((Path(model_dir) / "models_export.json").read_text(
                encoding="utf-8"))
            md = recipe.get("module_dtypes", {})
            vae_dtype = (torch.float32 if str(md.get("vae_encoder", "")).endswith("float32")
                         else torch.float16)
            dev = torch.device("cuda")
            ref_dir = exp_dir / "component_refs"
            # 同配方组件对照：enc/dec 各两项（4 次 eager + 4 次 artifact）
            led.reserve("component", "a2_load_assets")
            cfg = PBcore.load_config()
            assets = PBcore.Assets(cfg, dev)
            enc_e = VAEEncode(assets.vae.to(vae_dtype), vae_dtype, torch.float16).to(dev).eval()
            dec_e = Decode(assets.dec.to(vae_dtype), assets.vae.config.scaling_factor,
                           vae_dtype, torch.float32).to(dev).eval()
            art = torch.jit.load(str(mp), map_location=dev).eval()
            checks = {}
            plan = [("enc_512", lambda: enc_e(ins("enc_512")), lambda: art.encode(ins("enc_512"), 512)),
                    ("enc_1024", lambda: enc_e(ins("enc_1024")), lambda: art.encode(ins("enc_1024"), 1024)),
                    ("dec_64", lambda: dec_e(ins("dec_64")), lambda: art.decode(ins("dec_64"), 64)),
                    ("dec_128", lambda: dec_e(ins("dec_128")), lambda: art.decode(ins("dec_128"), 128))]

            def ins(name):
                t = torch.from_numpy(np.load(ref_dir / ("in_%s.npy" % name),
                                             allow_pickle=False))
                return t.to(dev).to(torch.float16)

            ok = True
            ck = exp_dir / "a2_component_checks"
            ck.mkdir(exist_ok=True)
            for name, fe, fa in plan:
                led.reserve("component", "a2_%s_eager" % name)
                with torch.no_grad():
                    e = fe().detach().float().cpu().numpy()
                led.reserve("component", "a2_%s_artifact" % name)
                with torch.no_grad():
                    a = fa().detach().float().cpu().numpy()
                np.save(ck / ("%s_eager.npy" % name), e)
                np.save(ck / ("%s_artifact.npy" % name), a)
                d = np.abs(a - e)
                rec = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                       "finite": bool(np.isfinite(a).all()),
                       "passed": bool(float(d.max()) <= tol["same_precision_max_abs"] and
                                      float(d.mean()) <= tol["same_precision_mean_abs"] and
                                      bool(np.isfinite(a).all()))}
                checks[name] = rec
                ok = ok and rec["passed"]
                print("[a2] %s max=%.3e mean=%.3e passed=%s"
                      % (name, rec["max_abs"], rec["mean_abs"], rec["passed"]), flush=True)
            del enc_e, dec_e, assets, art
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            (exp_dir / "a2_component.json").write_text(
                json.dumps({"schema": "DELIV01R-a2-component/1", "checks": checks,
                            "recipe_dtypes": md, "verdict": "PASS" if ok else "FAIL"},
                           indent=1, ensure_ascii=False), encoding="utf-8")
            if not ok:
                raise RuntimeError("A2 同配方组件对照未通过：%s"
                                   % {k: v["max_abs"] for k, v in checks.items()})
            readings = r_verify.run_verification(run_dir, led, prereg_path, model_dir,
                                                 tol, exp_dir)
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
            partial.update({"schema": "DELIV01R-a2-readings/1", "status": "FAIL",
                            "error": detail, "traceback": traceback.format_exc()[-3000:],
                            "ledger": led.summary(), "elapsed_s": round(led.elapsed_s(), 3)})
            rp.write_text(json.dumps(partial, indent=1, ensure_ascii=False), encoding="utf-8")
            print("[a2] FAIL %s" % detail, flush=True)
    finally:
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
