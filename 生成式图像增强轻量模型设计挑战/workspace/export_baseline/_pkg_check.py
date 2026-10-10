import os, json, glob
for d in [r"D:\models\pkg_TinyV11b", r"D:\models\pkg_TinyV16", r"D:\models\待定_解耦探测", r"D:\models\pkg_M24img_V9brun"]:
    print("==", d, os.path.isdir(d))
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d))[:14]:
        fp = os.path.join(d, f)
        print(f"   {'DIR ' if os.path.isdir(fp) else 'FILE'} {f} {os.path.getsize(fp)/2**20:.2f} MiB" if os.path.isfile(fp) else f"   DIR  {f}")
    for cand in ("test30_metrics.json", "metrics.json", "gate.json", "README.md", "README.txt", "record.json"):
        fp = os.path.join(d, cand)
        if os.path.isfile(fp):
            txt = open(fp, encoding="utf-8", errors="replace").read()
            print(f"   --- {cand} ---")
            print("   " + txt[:1200].replace("\n", "\n   "))
