import json, os, shutil, time
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
before = shutil.disk_usage("D:\\").free/2**30
# 归位最终包
src, dst = r"D:\models\opt11\Track2_TinyRefV3.zip", r"D:\models\upload\Track2_TinyRefV3.zip"
if os.path.exists(src): shutil.move(src, dst); print("staged:", dst)
# 删除被支配的中间包（V1 无 unsharp、V2 时延 17ms；内容可由 ckpt 重建）
for p, why in [(r"D:\models\upload\Track2_TinyRefV1.zip", "中间包：无 unsharp，已被 V3 支配（可重建）"),
               (r"D:\models\upload\Track2_TinyRefV2.zip", "中间包：48ch refiner 时延 17ms，已被 V3 支配（可重建）")]:
    if os.path.exists(p):
        s = os.path.getsize(p)/2**30; os.remove(p)
        log['actions'].append({'action': 'rm', 'detail': p + '（' + why + '）', 'freed_gib': round(s, 3)})
        print("removed", os.path.basename(p), "%.3f GiB" % s)
after = shutil.disk_usage("D:\\").free/2**30
log['free_after'] = round(after, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("D free %.2f -> %.2f GiB" % (before, after))
print("upload/:")
for f in sorted(os.listdir(r"D:\models\upload")):
    p = os.path.join(r"D:\models\upload", f)
    print("   %-32s %7.1f MiB" % (f, os.path.getsize(p)/2**20) if os.path.isfile(p) else "   "+f)
