import os, shutil, json, time
log = {'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'actions': []}
def note(a, detail, freed=0.0):
    log['actions'].append({'action': a, 'detail': detail, 'freed_gib': round(freed, 3)})
    print(f"   {a}: {detail} (freed {freed:.3f} GiB)")

def free():
    t, u, f = shutil.disk_usage("D:\\")
    return f / 2**30

print("D free before: %.2f GiB" % free())

# 1) V11b 包（同速但未经平台验证，V7gan 更快且已验证）→ 删除 zip，保留门禁日志
z = r"D:\models\opt11\Track2_OPT10img_V11brun.zip"
if os.path.exists(z):
    sz = os.path.getsize(z) / 2**30
    os.remove(z); note("delete_zip", z + "（V11b 包：同进程对照显示不如 V7gan，未选）", sz)

# 2) 门禁脚本的临时解压目录（可重建）
h = r"D:\models\_harness_zip_sim"
if os.path.isdir(h):
    sz = sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(h) for f in fs) / 2**30
    shutil.rmtree(h, ignore_errors=True); note("rmtree_scratch", h + "（门禁临时解压目录，脚本会自动重建）", sz)

# 3) 两个选定包归位到 upload/（移动，不占额外空间）
up = r"D:\models\upload"
os.makedirs(up, exist_ok=True)
for name in ("Track2_OPT10img_V4grun.zip", "Track2_OPT10img_V7ganrun.zip"):
    src = os.path.join(r"D:\models\opt11", name); dst = os.path.join(up, name)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst); note("stage_upload", dst)
    elif os.path.exists(dst):
        note("already_staged", dst)

log['free_after'] = round(free(), 3)
print("D free after: %.2f GiB" % log['free_after'])
print("upload/ 内容:")
for f in sorted(os.listdir(up)):
    p = os.path.join(up, f)
    print("   ", f, "%.1f MiB" % (os.path.getsize(p) / 2**20) if os.path.isfile(p) else "DIR")
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("[done] -> D:\\models\\opt11\\cleanup_log.json")
