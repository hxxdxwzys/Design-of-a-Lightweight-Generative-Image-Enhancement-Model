import json, os, shutil, time
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
def dsize(p):
    return sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(p) for f in fs) / 2**30 if os.path.isdir(p) else 0
before = shutil.disk_usage("D:\\").free / 2**30
targets = [
    (r"D:\models\pkg_TinyV11b\output_dir", "v11b 出图目录（同内容已在 upload/Track2_TinyV11b.zip 内；可用 render_tiny 4 分钟重建）"),
    (r"D:\models\pkg_TinyV18\output_dir",  "v18 出图目录（同内容已在 upload/Track2_TinyV18.zip 内）"),
    (r"D:\models\opt11\eval_v18_s25000",   "v18 test30 渲染（指标已落盘 test30_v18_s25000.json）"),
    (r"D:\models\opt11\eval_v19_s25000",   "v19 test30 渲染（指标已落盘 test30_v19_s25000.json）"),
]
for p, why in targets:
    if os.path.isdir(p):
        s = dsize(p); shutil.rmtree(p, ignore_errors=True)
        log['actions'].append({'action': 'rmtree', 'detail': p + '（' + why + '）', 'freed_gib': round(s, 3)})
        print("removed %-46s %6.3f GiB" % (os.path.basename(p), s))
after = shutil.disk_usage("D:\\").free / 2**30
log['free_after'] = round(after, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("D free: %.2f -> %.2f GiB | C free %.2f GiB" % (before, after, shutil.disk_usage("C:\\").free / 2**30))
print("保留指标:", [f for f in os.listdir(r"D:\models\opt11") if f.startswith("test30_")])
