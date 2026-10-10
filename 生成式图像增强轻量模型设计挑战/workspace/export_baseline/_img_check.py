import os
p = r"D:\models\opt10\output_dir"
print(p, "exists:", os.path.isdir(p))
if os.path.isdir(p):
    fs = sorted(os.listdir(p)); print("   n =", len(fs), fs[:3], fs[-2:])
    print("   size GiB = %.2f" % (sum(os.path.getsize(os.path.join(p, f)) for f in fs)/2**30))
for q in [r"D:\models\pkg_Tinyv21\model_dir", r"D:\models\pkg_TinyRefV4\model_dir", r"D:\models\pkg_TinyV7gan\model_dir"]:
    md = q if os.path.isdir(q) else q.replace("pkg_TinyV7gan", "pkg_TinyV7gan")
    print(md, os.path.isdir(md), os.listdir(md) if os.path.isdir(md) else "")
