import hashlib, os, zipfile
up = r"D:\models\upload"
for f in ("Track2_TinyRefV4.zip", "Track2_TinyRefV3.zip"):
    p = os.path.join(up, f)
    if not os.path.exists(p):
        print(f, "MISSING"); continue
    with zipfile.ZipFile(p) as zf:
        bad = zf.testzip(); names = zf.namelist()
        n = len([x for x in names if x.startswith("output_dir/") and x.endswith(".jpg")])
        md = sorted(x for x in names if x.startswith("model_dir/"))
        first = zf.getinfo("output_dir/case1.jpg").file_size
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print("%s | %.1f MiB | testzip=%s | jpg=%d | %s | case1=%.2fMB | sha256=%s" %
          (f, os.path.getsize(p)/2**20, "OK" if bad is None else bad, n, md, first/2**20, h[:20]))
print("gate:", open(r"D:\models\opt11\gate_TinyRefV4.txt", encoding="utf-8", errors="replace").read().strip().splitlines()[-1])
print("upload/:", sorted(os.listdir(up)))
