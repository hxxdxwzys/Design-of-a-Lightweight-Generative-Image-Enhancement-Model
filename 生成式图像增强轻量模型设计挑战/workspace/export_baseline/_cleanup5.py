import json, os, shutil
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
def dsize(p):
    return sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(p) for f in fs)/2**30 if os.path.isdir(p) else (os.path.getsize(p)/2**30 if os.path.isfile(p) else 0)
before = shutil.disk_usage("D:\\").free/2**30
items = [
    (r"D:\models\pkg_TinyRefV1\output_dir", "TinyRefV1 出图目录（同内容在 upload/Track2_TinyRefV1.zip 内）"),
    (r"D:\models\opt11\eval_v1ref", "v1ref test30 渲染（指标已落盘）"),
    (r"D:\models\opt11\eval_v21_ref", "refiner 试点 test30 渲染（指标已落盘）"),
    (r"D:\models\upload\Track2_TinyV11b.zip", "已被 v18/v19/v20b/v21/TinyRefV1 支配的旧合规包（已提交；可由 ckpt 逐字节重建）"),
    (r"D:\models\upload\Track2_TinyV18.zip", "同上（已提交；可重建）"),
]
for p, why in items:
    if os.path.exists(p):
        s = dsize(p)
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        log['actions'].append({'action': 'rmtree' if os.path.isdir(p) else 'rm', 'detail': p + '（' + why + '）/ 2026-09-20 磁盘告急清理', 'freed_gib': round(s, 3)})
        print("removed %-58s %6.3f GiB" % (os.path.basename(p), s))
after = shutil.disk_usage("D:\\").free/2**30
log['free_after'] = round(after, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("D free: %.2f -> %.2f GiB | C free %.2f GiB" % (before, after, shutil.disk_usage("C:\\").free/2**30))
print("upload:", sorted(os.listdir(r"D:\models\upload")))
