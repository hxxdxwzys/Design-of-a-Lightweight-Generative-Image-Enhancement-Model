import os, zipfile, json, glob
print("== D:\\models\\pkg_M24img_V9brun（历史解耦包）==")
d = r"D:\models\pkg_M24img_V9brun"
for r_, ds, fs in os.walk(d):
    lvl = r_.count(os.sep) - d.count(os.sep)
    if lvl <= 1:
        print("   ", os.path.relpath(r_, d), "dirs=", ds[:4], "files=", fs[:6], "(n=%d)" % len(fs))
z = r"D:\models\待定_解耦探测\Track2_M24img_V9brun.zip"
with zipfile.ZipFile(z) as zf:
    names = zf.namelist()
    print("\n== zip 内容 ==", len(names), "entries")
    for n in names[:8]:
        print("   ", n, zf.getinfo(n).file_size)
    print("    ...")
    for n in [x for x in names if not x.lower().endswith(".jpg")][:12]:
        print("   ", n, zf.getinfo(n).file_size)
print("\n== 构建/门禁脚本候选 ==")
for pat in ["*m24img*", "*package_submit*", "*preupload*", "*harness*zip*", "*make_zip*", "*pkg*"]:
    hits = glob.glob(r"D:\tmp_timing\\" + pat) + glob.glob(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline\\" + pat) + glob.glob(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\\" + pat)
    if hits:
        print(f"   {pat}: {hits}")
print("\n== pkg_TinyV4g / V7gan model_dir ==")
for p in [r"D:\models\pkg_TinyV4g", r"D:\models\pkg_TinyV7gan"]:
    md = os.path.join(p, "model_dir")
    print("   ", p, os.listdir(md) if os.path.isdir(md) else "MISSING",
          [os.path.getsize(os.path.join(md, f)) for f in os.listdir(md)] if os.path.isdir(md) else "")
