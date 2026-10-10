"""Two fresh processes, fixed 15-minute shared deadline, max 300 forward reservations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="New empty evidence directory outside the package")
    a = ap.parse_args()
    from selfcheck import verify_identity, sha
    verify_identity()
    lock = HERE / ".verification.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        out = Path(a.output).resolve()
        if out.is_relative_to(HERE):
            raise ValueError("Evidence output must be outside the immutable package")
        out.mkdir(parents=True, exist_ok=False)
        start = time.time()
        protocol = {"cases": ["case3", "case17"], "processes": 2, "call_cap": 300,
                    "wall_seconds": 900, "deadline_epoch": start+900, "reserved_calls": 0,
                    "criterion": "both reference JPEGs byte-identical in both processes; no tolerance fallback",
                    "failure_rule": "retain failed readings; stop after failed process; no automatic retry",
                    "manifest_sha256": sha(HERE/"manifest.json"),
                    "runner_constructor_reservation_per_process": 4}
        (out/"preregistration.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        ledger = out/"ledger.json"
        ledger.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        results = []
        for i in [1, 2]:
            command = [sys.executable, "-B", str(HERE/"selfcheck.py"), "--output", str(out/("process_"+str(i))), "--ledger", str(ledger)]
            remaining = start + 900-time.time()
            if remaining <= 0:
                raise TimeoutError("Cumulative deadline exhausted")
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
            with (out/("process_"+str(i)+".log")).open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                try:
                    code = child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
                    else:
                        child.kill()
                    child.wait()
                    code = 124
            results.append({"process": i, "exit_code": code})
            print(results[-1], flush=True)
            if code != 0:
                break
        success = len(results) == 2 and all(x["exit_code"] == 0 for x in results)
        summary = {"status": "PASS" if success else "FAIL", "processes": results,
                   "wall_s": time.time()-start, "ledger": json.loads(ledger.read_text(encoding="utf-8")),
                   "not_certified": ["all 100 outputs", "official environment", "official eligibility"]}
        (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0 if success else 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
