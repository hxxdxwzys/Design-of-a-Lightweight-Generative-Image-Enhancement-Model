# -*- coding: utf-8 -*-
"""CPU 故障注入替身：启动后不退出，用于验证监督 deadline 终止与退出确认。"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import r_ledger  # noqa: E402

run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) for p in sys.argv[1:6]]
led = r_ledger.Ledger(ledger_path)
led.acquire_lock('fake-sleep')
led.open_attempt('attempt_01', run_dir)
print('[fake-sleep] started', flush=True)
time.sleep(600)
