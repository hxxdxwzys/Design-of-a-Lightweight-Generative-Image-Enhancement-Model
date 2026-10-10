# -*- coding: utf-8 -*-
"""PIPE-01R 覆盖矩阵生成器：把 review 的 R01–R06 映射到具体测试名，并用隔离复跑日志核对。

不做“全部覆盖”的宣称：只核对该映射里的每个测试名确实出现在日志中且为 ok。
用法: D:/Python/python.exe make_coverage_matrix.py <isolated_test_log> <out_json>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MATRIX = {
    'PIPE-R01_dino_preprocess': [
        'TestDinoPreprocess.test_matches_reference_for_all_cases',
        'TestDinoPreprocess.test_gray_input_is_not_wrongly_saturated',
        'TestDinoPreprocess.test_old_wrong_order_would_differ',
        'TestDinoPreprocess.test_white_and_gradient_finite',
    ],
    'PIPE-R02_safe_load_offline': [
        'TestSafeLoading.test_no_weights_only_false_in_sources',
        'TestSafeLoading.test_no_torch_hub_calls_in_sources',
        'TestSafeLoading.test_lora_unsafe_payload_fails_closed',
        'TestSafeLoading.test_lora_missing_file_fails',
        'TestSafeLoading.test_missing_or_wrong_dino_never_goes_online',
        'TestSafeLoading.test_wrong_checkpoint_fails_strict',
    ],
    'PIPE-R03_identity_and_prereg': [
        'TestIdentityAndPrereg.test_implementation_fingerprint_real',
        'TestIdentityAndPrereg.test_asset_fingerprint_covers_vae_and_dino_source',
        'TestIdentityAndPrereg.test_identity_comparable_strips_entry_only',
        'TestDriverEndToEndCPU.test_identity_mismatch_never_passes',
        'TestDriverEndToEndCPU.test_prereg_mismatch_blocks_before_any_load',
        'TestDriverEndToEndCPU.test_driver_full_flow_passes_with_identical_entries',
    ],
    'PIPE-R04_budget_ledger': [
        'TestBudgetLedger.test_zero_quota_refuses',
        'TestBudgetLedger.test_reserve_is_persisted_before_call_and_not_refunded',
        'TestBudgetLedger.test_boundary_last_unit_allowed_then_refused',
        'TestBudgetLedger.test_cross_attempt_accumulates_and_interruption_counts',
        'TestBudgetLedger.test_new_process_cannot_reset_quota',
        'TestBudgetLedger.test_missing_ledger_without_init_refuses',
        'TestBudgetLedger.test_wall_stop',
        'TestBudgetLedger.test_concurrent_lock_refused',
        'TestBudgetLedger.test_disk_budget',
        'TestDriverEndToEndCPU.test_zero_quota_blocks_before_any_load',
        'TestDriverEndToEndCPU.test_nonempty_run_dir_is_refused',
    ],
    'PIPE-R05_nan_before_quantization': [
        'TestNaNBeforeQuantization.test_require_finite_blocks',
        'TestNaNBeforeQuantization.test_uint8_quantization_would_hide_nan_but_precheck_blocks',
        'TestNaNBeforeQuantization.test_infer_core_blocks_nan_latent',
    ],
    'PIPE-R06_timing_and_unique_artifacts': [
        'TestSharedCoreAndEntries.test_offline_records_separate_io_timing',
        'TestSharedCoreAndEntries.test_both_entries_call_same_core',
        'TestSharedCoreAndEntries.test_png_encoder_is_shared_and_pixel_identical',
        'TestDriverEndToEndCPU.test_driver_full_flow_passes_with_identical_entries',
    ],
    'PIPE_R08_generation_manifest': [
        'TestDriverEndToEndCPU.test_driver_full_flow_passes_with_identical_entries',
        'TestDriverEndToEndCPU.test_criteria_exceeded_stops_immediately',
    ],
    'PIPE_RA01_real_assembly_control_flow': [
        'TestRealAssetsAssembly.test_real_assets_init_control_flow',
    ],
    'PIPE_RA02_immediate_stop': [
        'TestImmediateStop.test_stops_after_first_call_failure_and_keeps_charge',
        'TestImmediateStop.test_identity_mismatch_blocks_before_b_first_forward',
        'TestImmediateStop.test_stops_at_first_threshold_exceeded',
    ],
    'PIPE_RA03_instance_release_order': [
        'TestInstanceLifecycle.test_a_released_before_b_constructed',
        'TestInstanceLifecycle.test_a_released_even_when_a_fails',
        'TestRealAssetsAssembly.test_real_assets_release_clears_modules',
    ],
    'PIPE_RA04_lock_timeout_finalize': [
        'TestLedgerHardening.test_exclusive_lock_and_stale_takeover',
        'TestLedgerHardening.test_pid_state_tristate_and_unknown_fails_closed',
        'TestLedgerHardening.test_check_startable_refuses_zero_quota',
        'TestLedgerHardening.test_check_startable_refuses_when_time_exhausted',
        'TestLedgerHardening.test_disk_is_cumulative_across_attempts',
        'TestLedgerHardening.test_end_attempt_records_disk_and_wall',
        'TestSupervisorHardWall.test_timeout_terminates_child',
        'TestSupervisorHardWall.test_child_within_deadline_completes',
    ],
    'PIPE_RA05_protocol_and_ledger_binding': [
        'TestProtocolBinding.test_accepts_matching_protocol',
        'TestProtocolBinding.test_rejects_crop_change_and_wrong_samples',
        'TestProtocolBinding.test_generator_freezes_eight_forwards_and_zero_warmups',
        'TestProtocolBinding.test_driver_blocks_on_protocol_mismatch_before_load',
        'TestProtocolBinding.test_ledger_binding_mismatch_blocks',
    ],
    'PIPE_RA06_shared_infer_timing_boundary': [
        'TestDriverEndToEndCPU.test_driver_full_flow_passes_with_identical_entries',
        'TestSharedCoreAndEntries.test_offline_records_separate_io_timing',
    ],
    'PIPE_RA07_manifest_hash_semantics': [
        'TestManifestSemantics.test_tensor_and_npy_hashes_are_separately_recomputable',
        'TestManifestSemantics.test_failed_call_marks_artifacts_not_produced',
    ],
    'PIPE_RB01_single_owner_control_chain': [
        'TestParentChildIntegration.test_normal_run_charges_exactly_eight_and_passes_all_four_gates',
        'TestParentChildIntegration.test_missing_prereg_and_protocol_mismatch_never_succeed',
    ],
    'PIPE_RB02_no_stale_ledger_overwrite': [
        'TestParentChildIntegration.test_supervisor_recovery_does_not_erase_child_reservations',
        'TestParentChildIntegration.test_second_forward_failure_charges_two_and_never_builds_b',
        'TestParentChildIntegration.test_timeout_confirms_child_exit_and_keeps_charge',
    ],
    'PIPE_RB03_strict_protocol_and_four_gate_verdict': [
        'TestParentChildIntegration.test_identity_mismatch_zero_b_forwards',
        'TestParentChildIntegration.test_normal_run_charges_exactly_eight_and_passes_all_four_gates',
    ],
    'retained_original_intent': [
        'TestConfigFrozen.test_frozen_config_loads_and_locks',
        'TestConfigFrozen.test_rejects_tampered_switches',
        'TestConfigFrozen.test_rejects_unknown_and_missing_keys',
        'TestConfigFrozen.test_tile_params_match_task_sheet',
        'TestConfigFrozen.test_tile_report_rejects_recipe_change',
        'TestConfigFrozen.test_args_ns_conflict_is_rejected',
        'TestSharedCoreAndEntries.test_prompt_must_be_empty',
        'TestSharedCoreAndEntries.test_model_dir_must_exist',
        'TestFailurePropagation.test_output_validation',
        'TestFailurePropagation.test_driver_refuses_without_prereg',
        'TestFailurePropagation.test_make_evidence_refuses_existing_run_dir_without_writing',
        'TestSeedAndShapes.test_local_seed_is_order_independent',
        'TestSeedAndShapes.test_pad_and_crop_for_non_multiple_shapes',
        'TestImportSafety.test_imports_do_not_pull_torch',
        'TestImportSafety.test_cli_help_does_not_load_model_or_cuda',
    ],
}


def main(argv):
    log = Path(argv[1])
    out = Path(argv[2])
    txt = log.read_text(encoding='utf-8', errors='replace')
    # 用机器可读行 TEST_OK <id>（run_tests_machine.py 产出）；unittest -v 会按宽度折行、且打印 docstring
    # 注意：unittest 的进度点会与自定义输出同行（".TEST_OK ..."），因此不锚定行首
    ok_short = {m.group(1).split('.')[-1]
                for m in re.finditer(r'TEST_OK (\S+)', txt)}
    ran = re.search(r'MACHINE_SUMMARY tests=(\d+) failures=(\d+) errors=(\d+)', txt)
    res = {'log': str(log),
           'ran_tests': int(ran.group(1)) if ran else None,
           'failures': int(ran.group(2)) if ran else None,
           'errors': int(ran.group(3)) if ran else None,
           'log_verdict': ('OK' if (ran and ran.group(2) == '0' and ran.group(3) == '0')
                           else 'NOT_OK'),
           'mapped': {}, 'unmapped_tests': [], 'missing_tests': []}
    mapped_short = set()
    for item, tests in MATRIX.items():
        rows = []
        for t in tests:
            short = t.split('.')[-1]
            mapped_short.add(short)
            rows.append({'test': t, 'in_log_ok': short in ok_short})
        res['mapped'][item] = {'tests': rows,
                               'all_present_ok': all(r['in_log_ok'] for r in rows)}
    res['unmapped_tests'] = sorted(n for n in ok_short if n not in mapped_short)
    res['missing_tests'] = sorted(t['test'] for item in res['mapped'].values()
                                  for t in item['tests'] if not t['in_log_ok'])
    res['note'] = ('手工映射（R 项 → 测试名），并用隔离复跑日志逐名核对；'
                   '不宣称“全部覆盖”，未映射测试列在 unmapped_tests。')
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('mapped_items=%d ran_tests=%s verdict=%s missing=%d unmapped=%d'
          % (len(MATRIX), res['ran_tests'], res['log_verdict'],
             len(res['missing_tests']), len(res['unmapped_tests'])))
    return 0 if not res['missing_tests'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
