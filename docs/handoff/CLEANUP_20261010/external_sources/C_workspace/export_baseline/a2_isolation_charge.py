# -*- coding: utf-8 -*-
"""C3 隔离预检的额度编排：先扣后跑（4 次 512 顶层调用额度），再执行 isolated_precheck.py。

用法: python a2_isolation_charge.py <run_dir> <zip_path> <ledger_path>
说明：隔离预检内部会在全新子进程里做 1 次 512 前向；按授权「至少为 100 图 + 4 次隔离保留调用」
      先预留 4 次（失败不退）；实际使用次数写入 readings 供核对。
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r_ledger  # noqa: E402
import r_verify  # noqa: E402


def main():
    run_dir, zip_path, ledger_path = [Path(p) for p in sys.argv[1:4]]
    led = r_ledger.Ledger.from_file(ledger_path)
    try:
        led.acquire_lock("C3 isolation precheck")
    except r_ledger.LockHeld as e:
        print("[iso] LOCK_HELD %s" % e, flush=True)
        return 2
    status, detail = "FAIL", ""
    try:
        model_p = Path(run_dir) / "model_dir" / "your_model.pt"
        led.open_attempt(os.environ.get("DELIV02_ATTEMPT_ID", "a2_isolation"), run_dir,
                         prereg_sha256=None,
                         source_sha256=r_verify.sha256_file(HERE / "a2_isolation_charge.py"),
                         model_sha256=(r_verify.sha256_file(model_p) if model_p.is_file()
                                       else None))
        for i in range(4):                      # 预留 4 次隔离额度（失败不退）
            led.reserve("top", "isolation_precheck_%d" % (i + 1))
        cp = subprocess.run([sys.executable, "-B", str(HERE / "isolated_precheck.py"),
                             str(run_dir), str(zip_path)], capture_output=True, text=True,
                            timeout=1800)
        print("[iso] rc=%d" % cp.returncode, flush=True)
        print((cp.stdout or "").strip()[-800:], flush=True)
        if cp.stderr:
            print("[iso][stderr] " + (cp.stderr or "").strip()[-600:], flush=True)
        rec = {"schema": "DELIV02-isolation-charge/1", "zip": str(zip_path),
               "reserved_top_calls": 4, "used_forward_calls": 1,
               "isolated_precheck_rc": cp.returncode}
        p = Path(run_dir) / "evidence" / "isolation_charge.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        status = "OK" if cp.returncode == 0 else "FAIL"
        detail = "rc=%d" % cp.returncode
    finally:
        try:
            led.close_attempt(status, note=detail)
        except Exception:
            pass
        led.release_lock()
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
