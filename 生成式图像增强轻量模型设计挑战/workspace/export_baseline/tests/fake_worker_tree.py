# -*- coding: utf-8 -*-
"""CPU 故障注入替身：worker 派生一个长睡孙进程后自身也长睡（验证整棵树被终止）。"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import r_ledger  # noqa: E402

run_dir, ledger_path, prereg_path, model_dir, exp_dir = [Path(p) for p in sys.argv[1:6]]
led = r_ledger.Ledger(ledger_path)
led.acquire_lock('fake-tree')
led.open_attempt('attempt_01', run_dir)
g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
(Path(exp_dir) / 'grandchild_pid.txt').write_text(str(g.pid), encoding='utf-8')
print('[fake-tree] grandchild=%d' % g.pid, flush=True)
time.sleep(600)
