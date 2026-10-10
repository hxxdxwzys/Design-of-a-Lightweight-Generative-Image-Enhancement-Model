import json, os, shutil
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
def dsize(p):
    return sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(p) for f in fs)/2**30 if os.path.isdir(p) else 0
before = shutil.disk_usage("D:\\").free/2**30
src, dst = r"D:\models\opt11\Track2_Tinyv21.zip", r"D:\models\upload\Track2_TinyV21.zip"
if os.path.exists(src): shutil.move(src, dst); print("staged:", dst)
for p, why in [(r"D:\models\pkg_Tinyv21\output_dir", "v21 出图目录（同内容在 upload/Track2_TinyV21.zip 内）"),
               (r"D:\models\opt11\eval_v21", "v21 test30 渲染（指标已落盘 test30_v21.json）")]:
    if os.path.isdir(p):
        s = dsize(p); shutil.rmtree(p, ignore_errors=True)
        log['actions'].append({'action': 'rmtree', 'detail': p + '（' + why + '）', 'freed_gib': round(s, 3)})
        print("removed %-34s %6.3f GiB" % (os.path.basename(p), s))
after = shutil.disk_usage("D:\\").free/2**30
log['free_after'] = round(after, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("D free: %.2f -> %.2f GiB | C free %.2f GiB" % (before, after, shutil.disk_usage("C:\\").free/2**30))
print("\nupload/:")
for f in sorted(os.listdir(r"D:\models\upload")):
    p = os.path.join(r"D:\models\upload", f)
    print("   %-34s %6.1f MiB" % (f, os.path.getsize(p)/2**20) if os.path.isfile(p) else "   " + f)
