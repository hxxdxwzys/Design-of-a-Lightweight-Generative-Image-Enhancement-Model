import os
for p in [r"D:\models\opt10\npy", r"D:\models\opt10\evidence"]:
    print("==", p)
    if os.path.isdir(p):
        for f in sorted(os.listdir(p))[:14]:
            fp = os.path.join(p, f); print("   ", f, os.path.getsize(fp) if os.path.isfile(fp) else "DIR")
