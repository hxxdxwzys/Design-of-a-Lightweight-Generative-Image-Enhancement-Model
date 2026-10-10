# -*- coding: utf-8 -*-
"""DELIV-01R A3 worker：分段诊断（≤20 次组件调用），只诊断、不签收。"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_mode  # noqa: E402
import r_ledger  # noqa: E402


def main():
    run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) for p in sys.argv[1:6]]
    flags = frozen_mode.apply_frozen_mode()
    (exp_dir / "a3_execution_flags.json").write_text(json.dumps(flags, indent=1),
                                                     encoding="utf-8")
    led = r_ledger.Ledger(ledger_path)
    try:
        led.acquire_lock("DELIV01R A3")
    except r_ledger.LockHeld as e:
        print("[a3] LOCK_HELD %s" % e, flush=True)
        return 2
    status, detail = "FAIL", ""
    try:
        led.open_attempt("attempt_a3", run_dir,
                         prereg_sha256=json.dumps(flags, sort_keys=True),
                         source_sha256=None, model_sha256=None)
        try:
            led.reserve("component", "a3_load_assets")
            for i in range(19):
                led.reserve("component", "a3_segment_forward_%d" % (i + 1))
            cp = subprocess.run([sys.executable, "-B", str(HERE / "a3_segment_probe.py"),
                                 str(exp_dir)], capture_output=True, text=True, timeout=900)
            print("[a3] rc=%d %s" % (cp.returncode, (cp.stdout or "").strip()[-300:]),
                  flush=True)
            if cp.returncode != 0:
                raise RuntimeError("A3 探针失败：%s" % ((cp.stderr or cp.stdout)[-400:]))
            status = "OK"
        except Exception as e:
            detail = "%s: %s" % (type(e).__name__, e)
            (exp_dir / "a3_failure.json").write_text(
                json.dumps({"error": detail, "traceback": traceback.format_exc()[-3000:],
                            "ledger": led.summary()}, indent=1, ensure_ascii=False),
                encoding="utf-8")
            print("[a3] FAIL %s" % detail, flush=True)
    finally:
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
