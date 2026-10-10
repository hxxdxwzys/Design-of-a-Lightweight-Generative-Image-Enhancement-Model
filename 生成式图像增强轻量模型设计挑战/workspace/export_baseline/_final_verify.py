import hashlib, os, zipfile, shutil
def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]
opt10_zip = r"D:\models\opt10\Track2_OPT10_fp16full_unsharp_v1.zip"
new = r"D:\models\upload\Track2_OPT10img_V4grun.zip"
with zipfile.ZipFile(opt10_zip) as a, zipfile.ZipFile(new) as b:
    names = [n for n in b.namelist() if n.startswith("output_dir/")]
    print("新包 output_dir 条目:", len(names))
    for probe in ("output_dir/case1.jpg", "output_dir/case54.jpg", "output_dir/case100.jpg"):
        ha = sha(a.read(probe)); hb = sha(b.read(probe))
        print(f"   {probe}: 已提交包 {ha} | 新包 {hb} -> {'一致' if ha == hb else '不一致'}")
    md = [n for n in b.namelist() if n.startswith("model_dir/")]
    print("新包 model_dir:", md, [b.getinfo(n).file_size for n in md])
print()
for z in ("Track2_OPT10img_V4grun.zip", "Track2_OPT10img_V7ganrun.zip"):
    p = os.path.join(r"D:\models\upload", z)
    with zipfile.ZipFile(p) as zf:
        bad = zf.testzip()
    print(f"   {z}: {os.path.getsize(p):,} B  testzip={'OK' if bad is None else bad}")
for d in ("C:\\", "D:\\"):
    print(f"   {d} free {shutil.disk_usage(d).free/2**30:.2f} GiB")
