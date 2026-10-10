import json, os, shutil
log = json.load(open(r"D:\models\opt11\cleanup_log.json", encoding="utf-8"))
def dsize(p):
    return sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(p) for f in fs)/2**30 if os.path.isdir(p) else (os.path.getsize(p)/2**30 if os.path.isfile(p) else 0)
before = shutil.disk_usage("D:\\").free/2**30
items = [
 (r"D:\models\opt11\Track2_TinyRefV6.zip", "r256c@10k 包：质量不合格（clipiqa 0.679），已撤出交接区，留档删除"),
 (r"D:\models\pkg_TinyRefV6\output_dir", "V6 出图目录"),
 (r"D:\models\pkg_TinyRefV4us0\output_dir", "V4us0 出图目录（zip 已提交，可重建）"),
 (r"D:\models\opt11\eval_v6", "V6 评测渲染（指标已落盘）"),
 (r"D:\models\opt11\eval_stu128", "学生@128 评测渲染（指标已落盘）"),
 (r"D:\models\refine_v1\in128", "r128 训练输入（需要时可 1.5 分钟重建；脚本 opt11_gen_data128.py 在库）"),
]
for p, why in items:
    if os.path.exists(p):
        s = dsize(p)
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        log['actions'].append({'action': 'rmtree' if os.path.isdir(p) else 'rm', 'detail': p + '（' + why + '）/ 2026-09-21 13:3x 收尾', 'freed_gib': round(s, 3)})
        print("removed %-40s %6.3f GiB" % (os.path.basename(p), s))
after = shutil.disk_usage("D:\\").free/2**30
log['free_after'] = round(after, 3)
open(r"D:\models\opt11\cleanup_log.json", "w", encoding="utf-8").write(json.dumps(log, indent=1, ensure_ascii=False))
print("D free %.2f -> %.2f GiB | C free %.2f GiB" % (before, after, shutil.disk_usage("C:\\").free/2**30))
print("upload/:", sorted(os.listdir(r"D:\models\upload")))
