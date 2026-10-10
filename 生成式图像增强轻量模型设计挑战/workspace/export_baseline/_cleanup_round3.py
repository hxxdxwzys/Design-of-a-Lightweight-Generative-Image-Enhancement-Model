import json, shutil, sys, time, ctypes
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LOG = Path(r"D:\models\opt11\cleanup_log.json")
def rd():
    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    return log if isinstance(log, list) else log.get("entries", [])
def wr(entries): LOG.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
def freespace():
    free = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p("D:\\"), None, None, ctypes.byref(free))
    return round(free.value/2**30, 2)
log = rd()
if not any(str(e.get("path","")).endswith("wave2_m0s7_s097") for e in log):
    log.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "path": r"D:\models\wave2_m0s7_s097",
                "files": None, "MiB": 3984.6, "reason": "wave2 scale-sweep 输出+打包模型（2026-09-15/16 实验，结论已落档；可从基座+LoRA 再生）",
                "deleted": True, "note": "事后补记：首次清理进程在删除后被杀，日志未写出", "agent": "DeepSeek"})
    wr(log); print("[补记] wave2_m0s7_s097", flush=True)
TARGETS = [
    (r"D:\models\hypir", "HYPIR 基线对照出图（已否决路线；结论在 docs/reference/新世界模型分数对应表.md）"),
    (r"D:\models\pipeline_baseline", "9/16-17 pipeline 基线中间件（.npy/.png，结论已落档）"),
    (r"D:\models\opt_probe", "OPT-10 探针模型副本（可从 C:/models/finals/preset/ckpts/VOSR_0.5B_os 再生；正式包在 D:\models\opt10）"),
]
for p, why in TARGETS:
    d = Path(p)
    if not d.exists(): print("skip", p, flush=True); continue
    files = [f for f in d.rglob("*") if f.is_file()]
    size = sum(f.stat().st_size for f in files)
    sample = [str(f.relative_to(d)) for f in files[:5]]
    shutil.rmtree(d, ignore_errors=True)
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "path": p, "files": len(files), "MiB": round(size/2**20,1),
           "reason": why, "deleted": not d.exists(), "sample": sample, "agent": "DeepSeek"}
    log = rd(); log.append(rec); wr(log)
    print("删除 %-20s %8.1f MiB files=%d ok=%s" % (d.name, size/2**20, len(files), not d.exists()), flush=True)
log = rd(); log.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "action": "freespace", "D_free_GiB": freespace(), "agent": "DeepSeek"}); wr(log)
print("D 可用: %.2f GiB" % freespace(), flush=True)
print("CLEAN_DONE", flush=True)