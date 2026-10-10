# -*- coding: utf-8 -*-
"""阶段 C 执行 + 读数合并（CPU 编排；GPU 前向在 c_fresh_worker.py 内，自带锁与 attempt）。

用法: python a2_run_phase_c.py <run_dir> <exp_dir> <ledger> <prereg> <model_dir>
前置：<exp_dir>/readings.json 存在且 status=OK、模型哈希与 model_dir 一致（由 C 自己复核）。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_mode  # noqa: E402
import r_verify  # noqa: E402


def main():
    run_dir, exp_dir, ledger, prereg, model_dir = [Path(p) for p in sys.argv[1:6]]
    frozen_mode.apply_frozen_mode()
    rd_p = exp_dir / "readings.json"
    rd = json.loads(rd_p.read_text(encoding="utf-8"))
    if rd.get("status") != "OK":
        print("[phaseC] 前置失败：readings.status=%s" % rd.get("status"), flush=True)
        return 2
    model_sha = r_verify.sha256_file(model_dir / "your_model.pt")
    if rd.get("model_sha256") != model_sha:
        print("[phaseC] 前置失败：模型哈希不一致", flush=True)
        return 2
    cp = subprocess.run([sys.executable, "-B", str(HERE / "c_fresh_worker.py"),
                         str(run_dir), str(ledger), str(prereg), str(model_dir), str(exp_dir)],
                        capture_output=True, text=True, timeout=2400, env=dict(os.environ))
    (exp_dir / "phase_c_console.txt").write_text(
        (cp.stdout or "") + "\n--- stderr ---\n" + (cp.stderr or ""), encoding="utf-8")
    print("[phaseC] rc=%d" % cp.returncode, flush=True)
    print(((cp.stdout or "") + (cp.stderr or "")).strip()[-600:], flush=True)
    pc = exp_dir / "phase_c.json"
    if pc.is_file():
        rd["phase_c"] = json.loads(pc.read_text(encoding="utf-8"))
    rd["phase_c_rc"] = cp.returncode
    rd["phase_c_tail"] = (cp.stdout or "").strip()[-400:]
    rd["status"] = "OK" if cp.returncode == 0 else "FAIL_PHASE_C"
    rd_p.write_text(json.dumps(rd, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if cp.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
