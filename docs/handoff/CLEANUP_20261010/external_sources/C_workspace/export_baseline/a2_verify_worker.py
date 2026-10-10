# -*- coding: utf-8 -*-
"""DELIV-01R A2 完整验证 worker：12 次完整成对（A/B、4K、迁移），不消耗组件额度。"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_mode  # noqa: E402
import r_ledger  # noqa: E402
import r_verify  # noqa: E402


def main():
    run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) for p in sys.argv[1:6]]
    flags = frozen_mode.apply_frozen_mode()
    (exp_dir / "verify_execution_flags.json").write_text(json.dumps(flags, indent=1),
                                                        encoding="utf-8")
    led = r_ledger.Ledger(ledger_path)
    try:
        led.acquire_lock("DELIV01R A2 verify")
    except r_ledger.LockHeld as e:
        print("[verify] LOCK_HELD %s" % e, flush=True)
        return 2
    status, detail = "FAIL", ""
    try:
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        mp = Path(model_dir) / "your_model.pt"
        led.open_attempt("attempt_verify", run_dir,
                         prereg_sha256=r_verify.sha256_file(prereg_path),
                         source_sha256=r_verify.sha256_file(HERE / "r_verify.py"),
                         model_sha256=r_verify.sha256_file(mp))
        try:
            # 顶层调用由 run_verification 在每次调用前 reserve（先扣后跑），此处不得重复预留
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
            partial.update({"schema": "DELIV01R-a2-readings/1", "status": "FAIL",
                            "error": detail, "traceback": traceback.format_exc()[-3000:],
                            "ledger": led.summary(), "elapsed_s": round(led.elapsed_s(), 3)})
            rp.write_text(json.dumps(partial, indent=1, ensure_ascii=False), encoding="utf-8")
            print("[verify] FAIL %s" % detail, flush=True)
    finally:
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
