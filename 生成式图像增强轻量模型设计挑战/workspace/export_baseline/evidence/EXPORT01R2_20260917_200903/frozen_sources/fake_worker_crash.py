# -*- coding: utf-8 -*-
"""CPU 故障注入替身：打开 attempt 后异常退出，不关闭 attempt、不释放锁（模拟崩溃）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import r_ledger  # noqa: E402

run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) for p in sys.argv[1:6]]
led = r_ledger.Ledger(ledger_path)
led.acquire_lock('fake-crash')
led.open_attempt('attempt_01', run_dir)
print('[fake-crash] dying', flush=True)
sys.exit(3)
