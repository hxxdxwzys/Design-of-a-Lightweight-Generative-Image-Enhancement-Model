import json, os, shutil
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
def dsize(p):
    return sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(p) for f in fs)/2**30 if os.path.isdir(p) else (os.path.getsize(p)/2**30 if os.path.isfile(p) else 0)
before = shutil.disk_usage("D:\\").free/2**30
items = [
 (r"D:\models\opt11\eval_w32", "w32 评测渲染（指标已落盘）"),
 (r"D:\models\opt11\Track2_TinyRefV5.zip", "w32@6k 包：无时延优势且质量未验（不进入 upload/，留档删除）"),
 (r"D:\models\pkg_TinyRefV5\output_dir", "V5 出图目录"),
 (r"D:\models\pkg_TinyRefV4\output_dir", "V4 出图目录（同内容在 upload/Track2_TinyRefV4.zip 内）"),
 (r"D:\models\pkg_prof_student_ref_us0", "探针目录"),
 (r"D:\models\pkg_prof_student_ref_us60", "探针目录"),
 (r"D:\models\pkg_probe_thin", "探针目录"),
]
for p, why in items:
    if os.path.exists(p):
        s = dsize(p)
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        log['actions'].append({'action': 'rmtree' if os.path.isdir(p) else 'rm', 'detail': p + '（' + why + '）/ 2026-09-20 17:0x 收尾清理', 'freed_gib': round(s, 3)})
        print("removed %-42s %6.3f GiB" % (os.path.basename(p), s))
after = shutil.disk_usage("D:\\").free/2**30
log['free_after'] = round(after, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("D free %.2f -> %.2f GiB" % (before, after))
open(r"D:\models\opt11\_add2.txt", "w", encoding="utf-8").write(open(r"D:\models\opt11\_add2_src.txt", encoding="utf-8").read())
p = r"C:\生成式图像增强轻量模型设计挑战\docs\handoff\P10007_20260920_细节合成器新路与最终版.md"
open(p, "a", encoding="utf-8").write(open(r"D:\models\opt11\_add2_src.txt", encoding="utf-8").read())
print("appended -> P10007")
