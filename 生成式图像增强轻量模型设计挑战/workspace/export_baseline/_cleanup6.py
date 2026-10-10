import json, os, shutil
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
def dsize(p):
    return sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(p) for f in fs)/2**30 if os.path.isdir(p) else (os.path.getsize(p)/2**30 if os.path.isfile(p) else 0)
before = shutil.disk_usage("D:\\").free/2**30
items = [
 (r"D:\models\opt11\Track2_TinyRefV2.zip", "中间包：48ch refiner 时延 17ms（被 V3 支配；可重建）"),
 (r"D:\models\pkg_TinyRefV2\output_dir", "V2 出图目录（zip 已删，需重渲染）"),
 (r"D:\models\pkg_TinyRefV3\output_dir", "V3 出图目录（同内容在 upload/Track2_TinyRefV3.zip 内）"),
 (r"D:\models\opt11\eval_gan1", "gan1 评测渲染（指标已落盘）"),
 (r"D:\models\opt11\eval_thin", "thin 评测渲染（指标已落盘）"),
 (r"D:\models\opt11\us_test", "unsharp60 评测渲染（指标已落盘）"),
 (r"D:\models\opt11\us_test100", "unsharp100 评测渲染（指标已落盘）"),
 (r"D:\models\upload\Track2_TinyV19.zip", "已被 V21/V3 支配的旧合规包（已提交，可重建）"),
 (r"D:\models\upload\Track2_TinyV20b.zip", "同上（已提交，可重建）"),
 (r"D:\models\pkg_probe_us0", "探针目录"),
 (r"D:\models\pkg_probe_us60", "探针目录"),
]
for p, why in items:
    if os.path.exists(p):
        s = dsize(p)
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        log['actions'].append({'action': 'rmtree' if os.path.isdir(p) else 'rm', 'detail': p + '（' + why + '）/ 2026-09-20 15:00 磁盘告急', 'freed_gib': round(s, 3)})
        print("removed %-52s %6.3f GiB" % (os.path.basename(p), s))
after = shutil.disk_usage("D:\\").free/2**30
log['free_after'] = round(after, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("D free %.2f -> %.2f GiB | C free %.2f GiB" % (before, after, shutil.disk_usage("C:\\").free/2**30))
