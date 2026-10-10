import os
cands = {
 "pkg_Tinyv21": r"D:\models\pkg_Tinyv21\model_dir",
 "pkg_TinyRefV3": r"D:\models\pkg_TinyRefV3\model_dir",
 "pkg_TinyRefV4": r"D:\models\pkg_TinyRefV4\model_dir",
 "pkg_TinyRefV5": r"D:\models\pkg_TinyRefV5\model_dir",
 "pkg_TinyRefV4us0": r"D:\models\pkg_TinyRefV4us0\model_dir",
 "pkg_Tinyv20b": r"D:\models\pkg_Tinyv20b\model_dir",
}
for k, p in cands.items():
    print("%-18s %s  %s" % (k, "OK " if os.path.isdir(p) else "MISS", os.listdir(p) if os.path.isdir(p) else ""))
print()
imgs = {
 "M24_img(提交件)": r"D:\models\pkg_M24img_V9brun\output_dir",
 "OPT10_img(提交件)": r"D:\models\opt10\output_dir",
}
for k, p in imgs.items():
    if os.path.isdir(p):
        fs = sorted(os.listdir(p)); print("%-18s OK  n=%d  %s" % (k, len(fs), fs[:2]))
    else:
        print("%-18s MISS" % k)
