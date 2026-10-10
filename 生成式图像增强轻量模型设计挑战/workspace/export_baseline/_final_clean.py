import json, os, shutil, zipfile, hashlib
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
before = shutil.disk_usage("D:\\").free/2**30
# 1) 按用户指示把解耦件移出交接区（用户已排除该路线）
for f in ("Track2_OPT10img_V4grun.zip", "Track2_OPT10img_V7ganrun.zip", "Track2_OPT10img_V21run.zip"):
    p = os.path.join(r"D:\models\upload", f)
    if os.path.exists(p):
        s = os.path.getsize(p)/2**30; os.remove(p)
        log['actions'].append({'action': 'rm', 'detail': p + '（用户指示「排除解耦件」，移出交接区；两件已提交、记录在台账，第三件未提交）/ 2026-09-20 16:3x', 'freed_gib': round(s, 3)})
        print("removed", f, "%.3f GiB" % s)
# 2) 最终合规包完整性核验
for f in ("Track2_TinyRefV4.zip", "Track2_TinyRefV3.zip"):
    p = os.path.join(r"D:\models\upload", f)
    if not os.path.exists(p): continue
    with zipfile.ZipFile(p) as zf:
        bad = zf.testzip()
        names = zf.namelist()
        n_jpg = len([n for n in names if n.startswith("output_dir/") and n.endswith(".jpg")])
        md = sorted(n for n in names if n.startswith("model_dir/"))
        ho = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    print("%s: %.1f MiB | testzip=%s | output_dir jpg=%d | model_dir=%s | sha256[:16]=%s" %
          (f, os.path.getsize(p)/2**20, "OK" if bad is None else bad, n_jpg, md, ho))
after = shutil.disk_usage("D:\\").free/2**30
log['free_after'] = round(after, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("D free %.2f -> %.2f GiB" % (before, after))
print("upload/:", sorted(os.listdir(r"D:\models\upload")))
