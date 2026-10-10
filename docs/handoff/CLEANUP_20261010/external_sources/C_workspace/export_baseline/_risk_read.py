import os
d = r"D:\models\待定_解耦探测"
for f in sorted(os.listdir(d)):
    print(repr(f), os.path.getsize(os.path.join(d, f)))
    if f.lower().endswith(".md"):
        print("---- content ----")
        print(open(os.path.join(d, f), encoding="utf-8", errors="replace").read()[:2500])
