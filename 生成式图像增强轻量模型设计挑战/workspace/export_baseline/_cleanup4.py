import json, os, shutil
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
def dsize(p):
    return sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(p) for f in fs)/2**30 if os.path.isdir(p) else 0
before = shutil.disk_usage("D:\\").free/2**30
print("== pkg_* 目录占用 ==")
for e in sorted(os.listdir(r"D:\models")):
    p = os.path.join(r"D:\models", e)
    if e.startswith("pkg_") and os.path.isdir(p):
        print("   %-22s %.3f GiB" % (e, dsize(p)))
for p, why in [(r"D:\models\pkg_Tinyv19\output_dir", "v19 出图目录（同内容在 upload/Track2_TinyV19.zip 内）")]:
    if os.path.isdir(p):
        s = dsize(p); shutil.rmtree(p, ignore_errors=True)
        log['actions'].append({'action': 'rmtree', 'detail': p + '（' + why + '）', 'freed_gib': round(s, 3)})
        print("removed", p, "%.3f GiB" % s)
after = shutil.disk_usage("D:\\").free/2**30
log['free_after'] = round(after, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("D free: %.2f -> %.2f GiB | C free %.2f GiB" % (before, after, shutil.disk_usage("C:\\").free/2**30))
