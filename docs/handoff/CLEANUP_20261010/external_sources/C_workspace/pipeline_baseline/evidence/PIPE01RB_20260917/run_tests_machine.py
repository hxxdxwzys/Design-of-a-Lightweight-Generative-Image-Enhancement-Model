# -*- coding: utf-8 -*-
"""PIPE-01R 证据用测试执行器：除了常规输出，还打印机器可读的 TEST_OK/TEST_FAIL/TEST_ERROR 行。

用法: D:/Python/python.exe run_tests_machine.py <tests_dir>
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


class Recorder(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        print('TEST_OK %s' % test.id(), flush=True)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        print('TEST_FAIL %s' % test.id(), flush=True)

    def addError(self, test, err):
        super().addError(test, err)
        print('TEST_ERROR %s' % test.id(), flush=True)


def main(argv):
    start = Path(argv[1]).resolve()
    loader = unittest.TestLoader()
    suite = loader.discover(str(start), pattern='test_pipe01_cpu.py')
    res = unittest.TextTestRunner(verbosity=1, resultclass=Recorder).run(suite)
    print('MACHINE_SUMMARY tests=%d failures=%d errors=%d skipped=%d'
          % (res.testsRun, len(res.failures), len(res.errors), len(res.skipped)), flush=True)
    return 0 if res.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
