import json, os, shutil, time
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
def dsize(p):
    return sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(p) for f in fs)/2**30 if os.path.isdir(p) else 0
before = shutil.disk_usage("D:\\").free/2**30
# 1) 归位 zip
src = r"D:\models\opt11\Track2_Tinyv20b.zip"; dst = r"D:\models\upload\Track2_TinyV20b.zip"
if os.path.exists(src):
    shutil.move(src, dst); print("staged:", dst)
# 2) 清理：出包后的 output_dir（zip 内已含同图）+ 评测渲染（指标已落盘）
for p, why in [(r"D:\models\pkg_Tinyv20b\output_dir", "v20b 出图目录（同内容在 upload/Track2_TinyV20b.zip 内）"),
               (r"D:\models\opt11\eval_v20b", "v20b test30 渲染（指标已落盘 test30_v20b.json）")]:
    if os.path.isdir(p):
        s = dsize(p); shutil.rmtree(p, ignore_errors=True)
        log['actions'].append({'action': 'rmtree', 'detail': p + '（' + why + '）', 'freed_gib': round(s, 3)})
        print("removed %-40s %6.3f GiB" % (os.path.basename(p), s))
after = shutil.disk_usage("D:\\").free/2**30
log['free_after'] = round(after, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("D free: %.2f -> %.2f GiB" % (before, after))
print("upload/:", sorted(os.listdir(r"D:\models\upload")))
