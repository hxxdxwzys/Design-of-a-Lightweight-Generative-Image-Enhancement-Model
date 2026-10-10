import os, json, shutil, time
for d in [r"D:\tmp_timing\harness_test", r"D:\tmp_timing\harness_sim2", r"D:\tmp_timing\pack_tmp"]:
    print("==", d)
    if os.path.isdir(d):
        for e in sorted(os.listdir(d))[:10]:
            print("   ", e, "(dir)" if os.path.isdir(os.path.join(d, e)) else "%.1f MiB" % (os.path.getsize(os.path.join(d, e)) / 2**20))
print("\nD free before: %.2f GiB" % (shutil.disk_usage("D:\\").free / 2**30))
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
for d in [r"D:\tmp_timing\harness_test", r"D:\tmp_timing\harness_sim2", r"D:\tmp_timing\pack_tmp"]:
    if os.path.isdir(d):
        sz = sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(d) for f in fs) / 2**30
        shutil.rmtree(d, ignore_errors=True)
        log['actions'].append({'action': 'rmtree_scratch', 'detail': d + '（历史 harness 模拟临时解压目录）', 'freed_gib': round(sz, 3)})
        print("   removed %s (%.2f GiB)" % (d, sz))
free = shutil.disk_usage("D:\\").free / 2**30
log['free_after'] = round(free, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("D free after: %.2f GiB" % free)
c = shutil.disk_usage("C:\\").free / 2**30
print("C free: %.2f GiB" % c)
