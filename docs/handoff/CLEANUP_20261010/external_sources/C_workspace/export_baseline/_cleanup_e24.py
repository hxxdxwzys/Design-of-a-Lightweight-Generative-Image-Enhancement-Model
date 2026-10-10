import json, os, shutil, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LOG = Path(r"D:\models\opt11\cleanup_log.json")
log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
TARGETS = [
    (r"D:\models\wave2_m0s7_s097", "wave2 scale-sweep 输出+打包模型（2026-09-15/16 实验，结论已入 score_ledger/文档；可从基座+LoRA 再生）"),
    (r"D:\models\hypir", "HYPIR 基线对照出图（已否决路线；结论在 docs/reference/新世界模型分数对应表.md）"),
    (r"D:\models\pipeline_baseline", "9/16-17 pipeline 基线中间件（.npy/.png，结论已落档）"),
    (r"D:\models\opt_probe", "OPT-10 探针模型副本（可从 C:/models/finals/preset/ckpts/VOSR_0.5B_os 再生；OPT-10 正式包在 D:\models\opt10）"),
]
def freespace():
    import ctypes
    free = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p("D:\\"), None, None, ctypes.byref(free))
    return round(free.value/2**30, 2)
before = freespace()
print("D 可用（清理前）: %.2f GiB" % before, flush=True)
for p, why in TARGETS:
    d = Path(p)
    if not d.exists(): print("skip(不存在)", p); continue
    files = [f for f in d.rglob("*") if f.is_file()]
    size = sum(f.stat().st_size for f in files)
    sample = [str(f.relative_to(d)) for f in files[:5]]
    shutil.rmtree(d, ignore_errors=True)
    ok = not d.exists()
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "path": p, "files": len(files),
           "MiB": round(size/2**20, 1), "reason": why, "deleted": ok, "sample": sample, "agent": "DeepSeek"}
    log.append(rec)
    print("删除 %-32s %8.1f MiB  files=%d  ok=%s" % (d.name, size/2**20, len(files), ok), flush=True)
after = freespace()
log.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "action": "freespace", "before_GiB": before, "after_GiB": after, "agent": "DeepSeek"})
LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
print("D 可用（清理后）: %.2f GiB" % after, flush=True)
print("cleanup logged ->", LOG)