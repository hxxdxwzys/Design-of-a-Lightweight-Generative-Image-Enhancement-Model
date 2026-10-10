import json, os, shutil
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
def dsize(p): return sum(os.path.getsize(os.path.join(r_,f)) for r_,_,fs in os.walk(p) for f in fs)/2**30 if os.path.isdir(p) else 0
before = shutil.disk_usage("D:\\").free/2**30
for e in sorted(os.listdir(r"D:\models\opt11")):
    p = os.path.join(r"D:\models\opt11", e)
    if os.path.isdir(p) and (e.startswith("e2_") or e.startswith("e5_")):
        s = dsize(p); shutil.rmtree(p, ignore_errors=True)
        log["actions"].append({"action":"rmtree","detail":p+"（评测渲染，proxy JSON 已保留）/ 2026-09-21 审计收尾","freed_gib":round(s,3)})
        print("removed", e, "%.3f GiB" % s)
after = shutil.disk_usage("D:\\").free/2**30
log["free_after"] = round(after,3)
open(r"D:\models\opt11\cleanup_log.json","w",encoding="utf-8").write(json.dumps(log,indent=1,ensure_ascii=False))
print("D free %.2f -> %.2f GiB" % (before, after))
open(r"C:\生成式图像增强轻量模型设计挑战\docs\reference\误判审计台账_20260921.md","a",encoding="utf-8").write(open(r"D:\models\opt11\_audit3.txt",encoding="utf-8").read())
open(r"C:\生成式图像增强轻量模型设计挑战\docs\progress.md","a",encoding="utf-8").write(open(r"D:\models\opt11\_s39.txt",encoding="utf-8").read())
print("appended 台账 E3-E5 + progress §39")