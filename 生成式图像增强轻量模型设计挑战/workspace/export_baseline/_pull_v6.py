import json, os, shutil
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
p = r"D:\models\upload\Track2_TinyRefV6.zip"
if os.path.exists(p):
    s = os.path.getsize(p)/2**30; os.remove(p)
    log['actions'].append({'action': 'rm', 'detail': p + '（r256c@10k：时延更快但 clipiqa 0.679 << V4 的 0.8046 ⇒ 预测 52.8 < 59.0，质量不合格，撤出交接区避免误交）/ 2026-09-20 17:5x', 'freed_gib': round(s, 3)})
    print("removed V6 from upload/  %.2f GiB" % s)
after = shutil.disk_usage("D:\\").free/2**30
log['free_after'] = round(after, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("upload/:", sorted(os.listdir(r"D:\models\upload")))
print("D free %.2f GiB" % after)
