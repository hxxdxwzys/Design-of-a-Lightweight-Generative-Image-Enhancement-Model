import os, json, glob
# 候选 runner 包：记录里的空闲时延/加速比
for d in [r"D:\models\pkg_M24img_V4grun", r"D:\models\pkg_TinyV7gan", r"D:\models\pkg_TinyV2g", r"D:\models\pkg_TinyV9b"]:
    print("==", d, os.path.isdir(d))
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        fp = os.path.join(d, f)
        if os.path.isfile(fp) and f.endswith((".json", ".txt")):
            try:
                txt = open(fp, encoding="utf-8", errors="replace").read()
            except Exception as e:
                txt = str(e)
            print(f"   --- {f} ---")
            print("   " + txt[:700].replace("\n", "\n   "))
        elif os.path.isdir(fp):
            print("   DIR", f, len(os.listdir(fp)), "entries")
print()
print("== OPT-10 100 图 ==")
od = r"D:\models\opt10\output_dir"
if os.path.isdir(od):
    fs = sorted(os.listdir(od))
    print("   n =", len(fs), "first:", fs[:4], "last:", fs[-3:])
print("== D:\\models 下所有 pkg 大小 ==")
for p in sorted(glob.glob(r"D:\models\pkg_*")):
    sz = sum(os.path.getsize(os.path.join(r_, x)) for r_, _, xs in os.walk(p) for x in xs) / 2**30
    print(f"   {p}  {sz:.2f} GiB")
