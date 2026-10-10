# -*- coding: utf-8 -*-
"""EXPORT-01R CPU 门槛：隔离复跑全部 CPU 用例 + 控制面场景 + 覆盖清单 + 源码冻结快照。

用法: python make_r_cpu_gate.py <evidence_dir>
产出: <evidence_dir>/{cpu_gate.json, cpu_test_results.txt, coverage_matrix.json, frozen_sources/}
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = ["chain.py", "delivery_runner.py", "r_ledger.py", "r_verify.py", "r_build.py",
       "r_worker.py", "r_supervisor.py", "r_merge_probe.py", "r2_prepare.py",
       "r2_make_ledger.py", "r3_make_ledger.py", "r3_component_gate.py",
       "make_r_prereg.py", "make_old_budget_errata.py",
       "export_config.json"]
TEST_FILES = ["test_wavelet_cpu.py", "test_delivery_cpu.py", "test_verify_logic_cpu.py",
              "test_export_r01_r03_cpu.py", "test_export_r_control_cpu.py",
              "test_r_pid_probe_cpu.py", "test_export_r2_cpu.py", "test_export_r3_cpu.py",
              "r_control_scenarios.py", "fake_worker_ok.py", "fake_worker_crash.py",
              "fake_worker_sleep.py", "fake_worker_tree.py"]
COVERAGE = {
    "EXP-R01": ["TestExpR01Fusion.test_zero_velocity_preserves_noise",
                "TestExpR01Fusion.test_const_and_varying_velocity_match_independent_formula",
                "TestExpR01Fusion.test_old_buggy_formula_differs"],
    "EXP-R02": ["r_control_scenarios:happy", "r_control_scenarios:crash",
                "TestSupervisorFaultInjection.test_happy_path_all_true",
                "TestSupervisorFaultInjection.test_crash_leaves_ledger_inconsistent"],
    "EXP-R03": ["TestExpR03NonFinite.test_nan_inf_rejected_before_clamp_and_quantization",
                "TestExpR03NonFinite.test_wavelet_would_mask_nonfinite"],
    "EXP-R04": ["TestLedger.test_reserve_is_pre_charged_and_non_refundable",
                "TestLedger.test_limits_enforced", "TestLedger.test_wall_clock_enforced",
                "TestLedger.test_lock_not_auto_taken_over",
                "TestLedger.test_cross_process_accumulation",
                "TestPidAlive.test_self_probe_is_safe",
                "r_control_scenarios:timeout", "r_control_scenarios:prelock"],
    "EXP-R05": ["r_merge_probe.py（独立加载 fp32 父实例对照，attempt 内执行）"],
    "EXP-R06": ["r_build.py 内容哈希/CastIn 记录", "r_verify.py 双方与重复 NPY 保存"],
    "EXP-R07": ["make_r_prereg.py 契约字段", "r_supervisor.py 产物核验段"],
    "EXP-R08": ["TestExpR08DecodeOrder.test_per_tile_clamp_then_blend",
                "TestExpR08DecodeOrder.test_nonfinite_still_intercepted"],
    "EXP-R09": ["TestExpR09PreflightIdentity.test_tampered_model_is_rejected_before_launch",
                "r2_prepare.py 新旧身份链", "frozen_sources 含 r_merge_probe/delivery_runner"],
    "EXP-R10": ["TestExpR10SupervisorEvaluation.test_complete_set_passes",
                "TestExpR10SupervisorEvaluation.test_missing_4k_duplicate_empty_and_hash_error_fail",
                "TestExpR11Criteria.test_required_set_detects_missing_duplicate_empty",
                "TestExpR11Criteria.test_release_check_cpu_path",
                "r_control_scenarios:tree"],
    "EXP-R11": ["TestExpR11Criteria.test_migration_over_limit_blocks",
                "r_merge_probe.py 判据段（merge max/mean/finite，超限停并保留两份输出）"],
    "EXP-R13": ["TestExpR13TimestepDevice.test_patched_matches_original_formula",
                "TestExpR13TimestepDevice.test_freqs_built_on_input_device",
                "TestExpR13TimestepDevice.test_trace_save_load_runs_after_patch",
                "TestExpR13TimestepDevice.test_graph_static_audit_no_cpu_arange_in_patched_source",
                "r3_component_gate.py check（保存后全新进程组件门槛，attempt 内执行）"],
    "EXP-R14": ["TestExpR14RealPath.test_required_set_ignores_non_dict_phaseA_values",
                "TestExpR14RealPath.test_partial_readings_persist_on_failure_path",
                "r_verify 逐次 flush + 失败保留 partial（真实驱动路径）"],
    "EXP-R15": ["r3_make_ledger.py 旧账关联（旧 5 top/2 component 扣款保留）",
                "r_build.py 新 SHA/new bundle 写入 bundle_manifest"],
}


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main():
    ev = Path(sys.argv[1]).resolve()
    ev.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="export01r_gate_"))
    pkg = work / "export_baseline"
    shutil.copytree(HERE, pkg, ignore=shutil.ignore_patterns("evidence", "__pycache__",
                                                             "_*", "*.npy", "*.pt"))
    # 只读兄弟目录：测试通过 _PB 定位 pipeline_baseline.core，隔离副本必须保持同级布局
    pb_src = HERE.parent / "pipeline_baseline"
    if pb_src.is_dir():
        shutil.copytree(pb_src, work / "pipeline_baseline",
                        ignore=shutil.ignore_patterns("evidence", "__pycache__", "*.npy", "*.pt"))
    gate = {"schema": "EXPORT01R-cpu-gate/1",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "isolated_dir": str(pkg), "coverage": COVERAGE, "checks": {}}
    # 1) 全量 CPU 用例
    rc1, out1 = run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests",
                     "-p", "test_*.py"], pkg)
    (ev / "cpu_test_results.txt").write_text(out1, encoding="utf-8")
    # 2) 机器可读计数
    rc2, out2 = run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests",
                     "-p", "test_*.py", "-v"], pkg)
    (ev / "cpu_test_verbose.txt").write_text(out2, encoding="utf-8")
    ran = 0
    for line in out2.splitlines():
        if line.startswith("Ran ") and " test" in line:
            ran = int(line.split()[1])
    # 3) 控制面场景（真子进程故障注入）
    scen_json = ev / "control_scenarios.json"
    rc3, out3 = run([sys.executable, "-B", str(pkg / "tests" / "r_control_scenarios.py"),
                     str(scen_json)], pkg)
    scen = json.loads(scen_json.read_text(encoding="utf-8")) if scen_json.is_file() else {}
    # 4) 冻结快照与哈希
    fz = ev / "frozen_sources"
    fz.mkdir(exist_ok=True)
    hashes = {}
    for rel in SRC + ["tests/" + t for t in TEST_FILES]:
        p = HERE / rel
        if not p.is_file():
            hashes[rel] = None
            continue
        tgt = fz / Path(rel).name
        shutil.copyfile(p, tgt)
        hashes[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size,
                       "frozen_copy": tgt.name, "copy_sha256": sha256_file(tgt)}
    gate["source_hashes_frozen"] = hashes
    missing_cov = [k for k, v in COVERAGE.items() if not v]
    gate["checks"] = {
        "suite_exit_0": rc1 == 0,
        "verbose_exit_0": rc2 == 0,
        "no_failures": "FAILED" not in out2,
        "n_tests": ran,
        "n_tests_at_least_45": ran >= 45,
        "control_scenarios_exit_0": rc3 == 0,
        "control_scenarios_all_pass": bool(scen.get("all_pass")),
        "coverage_entries_present": not missing_cov,
        "all_sources_frozen": all(v is not None for v in hashes.values()),
    }
    gate["open_issues"] = ["EXP-R05 的合并对照在 attempt 内以独立进程执行，CPU 侧只冻结脚本"]
    ok = all([gate["checks"]["suite_exit_0"], gate["checks"]["verbose_exit_0"],
              gate["checks"]["no_failures"], gate["checks"]["n_tests_at_least_45"],
              gate["checks"]["control_scenarios_exit_0"],
              gate["checks"]["control_scenarios_all_pass"],
              gate["checks"]["coverage_entries_present"],
              gate["checks"]["all_sources_frozen"]])
    gate["verdict"] = "READY_FOR_EXPORT01R_ATTEMPT_01" if ok else "NOT_READY"
    (ev / "cpu_gate.json").write_text(json.dumps(gate, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    (ev / "coverage_matrix.json").write_text(json.dumps(COVERAGE, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    print(json.dumps({"verdict": gate["verdict"], "checks": gate["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
