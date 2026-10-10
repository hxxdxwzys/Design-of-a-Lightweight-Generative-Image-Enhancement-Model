# -*- coding: utf-8 -*-
"""EXP-R04 补充：pid 存活探测必须安全（Windows 上 os.kill(pid,0) 会真的终止目标进程）。"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
EB = HERE.parent
sys.path.insert(0, str(EB))

import r_ledger  # noqa: E402


class TestPidAlive(unittest.TestCase):
    def test_self_probe_is_safe(self):
        # 若实现退回 os.kill(pid, 0)，本进程会被静默终止，测试进程直接消失
        self.assertTrue(r_ledger._pid_alive(os.getpid()))

    def test_dead_pid_is_false(self):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        self.assertFalse(r_ledger._pid_alive(p.pid))

    def test_live_child_is_true(self):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        try:
            self.assertTrue(r_ledger._pid_alive(p.pid))
        finally:
            p.kill()
            p.wait()

    def test_invalid_pid(self):
        self.assertFalse(r_ledger._pid_alive(0))
        self.assertFalse(r_ledger._pid_alive(-1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
