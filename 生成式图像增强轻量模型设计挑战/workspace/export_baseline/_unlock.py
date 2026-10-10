import json, os, sys
from pathlib import Path
sys.path.insert(0, r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline")
import r_ledger
lp = Path(r"D:\models\opt11\ledger_opt11.json.lock")
info = json.loads(lp.read_text(encoding="utf-8")) if lp.is_file() else None
print("lock:", info)
if info:
    alive = r_ledger._pid_alive(int(info.get("pid") or -1))
    print("owner alive:", alive)
    if not alive:
        lp.unlink()
        print("stale lock removed:", not lp.exists())
        led = r_ledger.Ledger.from_file(r"D:\models\opt11\ledger_opt11.json")
        led.note("opt11_modes_frozen", "清理陈旧锁 pid=%s（进程已死，脚本自愈）" % info.get("pid"))
        print("summary:", json.dumps(led.summary(), ensure_ascii=False)[:400])
