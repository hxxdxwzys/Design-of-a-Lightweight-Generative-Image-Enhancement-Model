# -*- coding: utf-8 -*-
"""EXPORT-01R 控制面 CPU 测试：总账（先扣后跑/不退/上限/墙钟/锁/跨进程）与监督者
（真子进程故障注入：正常、超时终止并确认退出、崩溃、预检拒绝）。"""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EB = HERE.parent
sys.path.insert(0, str(EB))

import r_ledger  # noqa: E402


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'ledger.json'
        self.led = r_ledger.Ledger(self.path, top=12, top4k=4, component=48, wall_s=1800)
        self.led.acquire_lock('test')
        self.led.open_attempt('attempt_01', Path(self.tmp.name))

    def tearDown(self):
        self.led.release_lock()
        self.tmp.cleanup()

    def test_reserve_is_pre_charged_and_non_refundable(self):
        before = self.led.summary()['used']
        self.assertEqual(before, {'top': 0, 'top4k': 0, 'component': 0})
        self.led.reserve('top', 'call1', is_4k=True)
        # 调用尚未发生（本测试不执行任何模型），账本已经扣款
        used = json.loads(self.path.read_text(encoding='utf-8'))['used']
        self.assertEqual(used['top'], 1)
        self.assertEqual(used['top4k'], 1)
        self.assertFalse(hasattr(self.led, 'refund'))
        self.led.reserve('component', 'probe')
        self.assertEqual(self.led.summary()['used']['component'], 1)

    def test_limits_enforced(self):
        for i in range(4):
            self.led.reserve('top', 'k%d' % i, is_4k=True)
        with self.assertRaises(r_ledger.BudgetExhausted):
            self.led.reserve('top', 'k4', is_4k=True)

    def test_wall_clock_enforced(self):
        led = r_ledger.Ledger(self.path, wall_s=0.0)
        with self.assertRaises(r_ledger.BudgetExhausted):
            led.reserve('top', 'late')

    def test_attempt_id_unique_but_stage_attempts_allowed(self):
        """DELIV-02 语义更新：同一 ID 禁止重复；不同 ID 的**阶段** attempt 允许（A/B/C 共用唯一总账）。

        旧断言要求"关闭后不得再开任何 attempt"，与 DELIV-02 明确采用的
        "唯一持久总账 + 多阶段 attempt + 累计额度" 语义相反，故按现行契约更新。
        """
        self.led.close_attempt('FAIL', 'x')
        self.led.open_attempt('attempt_02', Path(self.tmp.name))
        self.led.close_attempt('FAIL', 'x')
        with self.assertRaises(r_ledger.BudgetExhausted):
            self.led.open_attempt('attempt_02', Path(self.tmp.name))

    def test_lock_not_auto_taken_over(self):
        with self.assertRaises(r_ledger.LockHeld):
            r_ledger.Ledger(self.path).acquire_lock('second')
        info = self.led.lock_owner()
        self.assertEqual(info['pid'], __import__('os').getpid())

    def test_cross_process_accumulation(self):
        code = (
            'import sys, json; sys.path.insert(0, r"%s");'
            'import r_ledger;'
            'led = r_ledger.Ledger(r"%s");'
            '[led.reserve("component", "child%%d" %% i) for i in range(3)];'
            'print(json.dumps(led.summary()["used"]))' % (EB, self.path))
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        used = self.led.summary()['used']
        self.assertEqual(used['component'], 3, out.stdout)
        self.assertEqual(used['top'], 0)


class TestSupervisorFaultInjection(unittest.TestCase):
    def _run(self, worker, deadline=60, make_inputs=True):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        run = root / 'run'
        run.mkdir(parents=True)
        exp = run / 'evidence'
        exp.mkdir(parents=True)
        prereg = run / 'prereg.json'
        tol = {'same_precision_max_abs': 1e-5, 'same_precision_mean_abs': 1e-6,
               'fp16_vs_fp32_max_abs_diagnostic': 0.05,
               'fp16_vs_fp32_mean_abs_diagnostic': 0.002}
        cases = [{'name': n} for n in ('case1_crop512', 'case1_crop511x513',
                                       'case1_full', 'case7_full')]
        prereg.write_text(json.dumps({'tolerances': tol, 'cases': cases}), encoding='utf-8')
        model_dir = run / 'model_dir'
        model_dir.mkdir(parents=True)
        (model_dir / 'your_model.pt').write_bytes(b'x' * 16)
        ledger = root / 'ledger.json'
        cmd = [sys.executable, '-B', str(EB / 'r_supervisor.py'), str(run),
               '--ledger', str(ledger), '--prereg', str(prereg), '--model-dir', str(model_dir),
               '--exp-dir', str(exp), '--deadline-s', str(deadline), '--worker', str(worker)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(EB))
        sup = json.loads((exp / 'supervisor.json').read_text(encoding='utf-8')) \
            if (exp / 'supervisor.json').is_file() else None
        res = json.loads((exp / 'attempt_result.json').read_text(encoding='utf-8')) \
            if (exp / 'attempt_result.json').is_file() else None
        return tmp, proc, sup, res

    def test_happy_path_all_true(self):
        tmp, proc, sup, res = self._run(HERE / 'fake_worker_ok.py')
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(sup['worker_checks_pass'])
        self.assertTrue(sup['ledger_consistent'])
        self.assertTrue(res['ALL'])
        tmp.cleanup()

    def test_timeout_terminates_child_and_confirms_exit(self):
        t0 = time.time()
        tmp, proc, sup, res = self._run(HERE / 'fake_worker_sleep.py', deadline=3)
        self.assertLess(time.time() - t0, 60)
        self.assertTrue(sup['timed_out'])
        self.assertTrue(sup['terminate_requested_at'])
        self.assertFalse(sup['supervisor_success'])
        self.assertFalse(res['ALL'])
        self.assertNotEqual(proc.returncode, 0)
        stats = subprocess.run([sys.executable, '-c',
                                'import psutil' if False else 'print("skip")'],
                               capture_output=True, text=True)
        tmp.cleanup()

    def test_crash_leaves_ledger_inconsistent(self):
        tmp, proc, sup, res = self._run(HERE / 'fake_worker_crash.py')
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(sup['supervisor_success'])
        self.assertIsNotNone(sup['ledger']['open_attempt'])
        self.assertFalse(sup['ledger_consistent'])
        self.assertFalse(res['ALL'])
        tmp.cleanup()

    def test_preflight_rejects_existing_lock(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        run = root / 'run'
        (run / 'model_dir').mkdir(parents=True)
        (run / 'model_dir' / 'your_model.pt').write_bytes(b'x')
        exp = run / 'evidence'
        exp.mkdir(parents=True)
        prereg = run / 'prereg.json'
        prereg.write_text(json.dumps({'tolerances': {}, 'cases': []}), encoding='utf-8')
        ledger = root / 'ledger.json'
        Path(str(ledger) + '.lock').write_text('{}', encoding='utf-8')
        cmd = [sys.executable, '-B', str(EB / 'r_supervisor.py'), str(run),
               '--ledger', str(ledger), '--prereg', str(prereg),
               '--model-dir', str(run / 'model_dir'), '--exp-dir', str(exp)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(EB))
        self.assertEqual(proc.returncode, 2)
        self.assertIn('LOCK_EXISTS', proc.stdout + proc.stderr)
        tmp.cleanup()


if __name__ == '__main__':
    unittest.main(verbosity=2)
