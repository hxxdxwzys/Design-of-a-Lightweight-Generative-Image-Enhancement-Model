import os
rows = []
for root in (r"D:\models",):
    for e in sorted(os.listdir(root)):
        d = os.path.join(root, e, "output_dir")
        if os.path.isdir(d):
            fs = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
            if len(fs) >= 30:
                rows.append((e, len(fs), os.path.getsize(os.path.join(d, fs[0]))/2**20))
for n, k, mb in rows:
    print("%-26s jpg=%-4d first=%.1f MB" % (n, k, mb))
print("合计 %d 个可测目录" % len(rows))
