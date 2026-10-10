# -*- coding: utf-8 -*-
"""EXP-R04 控制面场景验证（独立脚本，避免 unittest 子进程语义干扰）。

四个场景全部用真实子进程与真实 r_supervisor.py 执行，结果写入 JSON：
  1 happy   : 完整走通锁/attempt/读数/产物 → ALL=True
  2 timeout : deadline 终止子进程并确认退出 → timed_out=True, supervisor_success=False
  3 crash   : worker 中途退出未关 attempt → ledger_consistent=False
  4 prelock : 预检发现已存在锁 → exit=2 LOCK_EXISTS（不启动子进程）
用法: python r_control_scenarios.py <out_json>
"""
import ctypes
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EB = HERE.parent


def pid_alive(pid):
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, int(pid))
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        return code.value == 259
    finally:
        k.CloseHandle(h)


def setup(root, model_bytes=b"x"):
    run = root / "run"
    (run / "model_dir").mkdir(parents=True, exist_ok=True)
    (run / "evidence").mkdir(parents=True, exist_ok=True)
    (run / "model_dir" / "your_model.pt").write_bytes(model_bytes)
    prereg = run / "prereg.json"
    prereg.write_text(json.dumps({
        "tolerances": {"same_precision_max_abs": 1e-5, "same_precision_mean_abs": 1e-6,
                       "fp16_vs_fp32_max_abs_diagnostic": 0.05,
                       "fp16_vs_fp32_mean_abs_diagnostic": 0.002},
        "cases": [{"name": n} for n in ("case1_crop512", "case1_crop511x513",
                                        "case1_full", "case7_full")]}), encoding="utf-8")
    return run, prereg


def run_sup(run, ledger, prereg, exp, worker, deadline):
    cmd = [sys.executable, "-B", str(EB / "r_supervisor.py"), str(run),
           "--ledger", str(ledger), "--prereg", str(prereg),
           "--model-dir", str(run / "model_dir"), "--exp-dir", str(exp),
           "--deadline-s", str(deadline), "--worker", str(worker)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(EB), timeout=300)
    sup = json.loads((exp / "supervisor.json").read_text(encoding="utf-8")) \
        if (exp / "supervisor.json").is_file() else None
    res = json.loads((exp / "attempt_result.json").read_text(encoding="utf-8")) \
        if (exp / "attempt_result.json").is_file() else None
    return p, sup, res


def scenario_happy(tmp):
    root = Path(tmp)
    run, prereg = setup(root)
    exp = run / "evidence"
    p, sup, res = run_sup(run, root / "ledger.json", prereg, exp,
                          HERE / "fake_worker_ok.py", 120)
    return {"exit": p.returncode, "worker_checks_pass": sup and sup["worker_checks_pass"],
            "ledger_consistent": sup and sup["ledger_consistent"], "ALL": res and res["ALL"],
            "pass": bool(p.returncode == 0 and res and res["ALL"])}


def scenario_timeout(tmp):
    root = Path(tmp)
    run, prereg = setup(root)
    exp = run / "evidence"
    t0 = time.time()
    p, sup, res = run_sup(run, root / "ledger.json", prereg, exp,
                          HERE / "fake_worker_sleep.py", 3)
    wall = time.time() - t0
    child = sup.get("child_pid") if sup else None
    child_gone = (not pid_alive(child)) if child else None
    return {"exit": p.returncode, "wall_s": round(wall, 2), "timed_out": sup and sup["timed_out"],
            "terminate_requested_at": sup and sup["terminate_requested_at"],
            "supervisor_success": sup and sup["supervisor_success"],
            "child_exit_confirmed": child_gone, "ALL": res and res["ALL"],
            "pass": bool(sup and sup["timed_out"] and not sup["supervisor_success"]
                         and child_gone and res and not res["ALL"] and wall < 120)}


def scenario_crash(tmp):
    root = Path(tmp)
    run, prereg = setup(root)
    exp = run / "evidence"
    p, sup, res = run_sup(run, root / "ledger.json", prereg, exp,
                          HERE / "fake_worker_crash.py", 120)
    led = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    return {"exit": p.returncode, "supervisor_success": sup and sup["supervisor_success"],
            "open_attempt": led.get("open_attempt") is not None,
            "ledger_consistent": sup and sup["ledger_consistent"], "ALL": res and res["ALL"],
            "pass": bool(p.returncode != 0 and sup and not sup["supervisor_success"]
                         and not sup["ledger_consistent"] and led.get("open_attempt")
                         and res and not res["ALL"])}


def scenario_prelock(tmp):
    root = Path(tmp)
    run, prereg = setup(root)
    exp = run / "evidence"
    ledger = root / "ledger.json"
    Path(str(ledger) + ".lock").write_text("{}", encoding="utf-8")
    p, sup, res = run_sup(run, ledger, prereg, exp, HERE / "fake_worker_ok.py", 60)
    out = (p.stdout or "") + (p.stderr or "")
    return {"exit": p.returncode, "mentions_lock": "LOCK_EXISTS" in out,
            "pass": bool(p.returncode == 2 and "LOCK_EXISTS" in out)}


def scenario_tree(tmp):
    """EXP-R10：worker 的孙进程在超时后也必须退出（整棵树终止）。"""
    root = Path(tmp)
    run, prereg = setup(root)
    exp = run / "evidence"
    p, sup, res = run_sup(run, root / "ledger.json", prereg, exp,
                          HERE / "fake_worker_tree.py", 3)
    gp_file = exp / "grandchild_pid.txt"
    gpid = int(gp_file.read_text(encoding="utf-8")) if gp_file.is_file() else None
    grandchild_gone = (not pid_alive(gpid)) if gpid else None
    tree = (sup or {}).get("process_tree", {})
    return {"exit": p.returncode, "timed_out": sup and sup["timed_out"],
            "grandchild_pid": gpid, "grandchild_gone": grandchild_gone,
            "process_tree": tree, "ALL": res and res["ALL"],
            "pass": bool(sup and sup["timed_out"] and grandchild_gone and
                         tree.get("clean") and res and not res["ALL"])}


def main():
    import signal
    try:                      # 本环境偶发外部 Ctrl+C：忽略之，避免场景中途被整体打断
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    results = {}
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("r_control_scenarios.json")

    def flush():
        out_path.write_text(json.dumps({"schema": "EXPORT01R-control-scenarios/1",
                                        "scenarios": results,
                                        "all_pass": all(v.get("pass") for v in results.values())},
                                       indent=1, ensure_ascii=False), encoding="utf-8")

    for name, fn in (("happy", scenario_happy), ("timeout", scenario_timeout),
                     ("crash", scenario_crash), ("prelock", scenario_prelock),
                     ("tree", scenario_tree)):
        with tempfile.TemporaryDirectory(prefix="r_scenario_",
                                          ignore_cleanup_errors=True) as tmp:
            try:
                results[name] = fn(tmp)
            except BaseException as e:
                results[name] = {"pass": False, "error": "%s: %s" % (type(e).__name__, e)}
            flush()
    verdict = {"schema": "EXPORT01R-control-scenarios/1", "scenarios": results,
               "all_pass": all(v.get("pass") for v in results.values())}
    out = out_path
    out.write_text(json.dumps(verdict, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"all_pass": verdict["all_pass"],
                      "names": {k: v.get("pass") for k, v in results.items()}}))


if __name__ == "__main__":
    main()
