import os
def show(p, label):
    if os.path.isdir(p):
        fs = sorted(os.listdir(p))
        print(f"{label}: DIR n={len(fs)} -> {fs[:18]}")
    elif os.path.exists(p):
        print(f"{label}: FILE {os.path.getsize(p):,} B")
    else:
        print(f"{label}: MISSING ({p})")

show(r"D:\models\opt01", "D:\\models\\opt01")
show(r"C:\models\finals\preset\ckpts\VOSR_0.5B_os", "base ckpt (C)")
show(r"C:\models\finals\preset\ckpts", "ckpts dir")
show(r"D:\models", "D:\\models")
show(r"C:\models", "C:\\models")
show(r"C:\models\exports", "C:\\models\\exports")

def dsize(p):
    tot = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try: tot += os.path.getsize(os.path.join(r, f))
            except Exception: pass
    return tot / 2**30
for p in [r"D:\models", r"C:\models", r"D:\tmp_timing"]:
    if os.path.isdir(p):
        print(f"size {p} = {dsize(p):.2f} GiB")
import shutil
for d in ["C:\\", "D:\\"]:
    t, u, f = shutil.disk_usage(d)
    print(f"{d} total {t/2**30:.1f} GiB free {f/2**30:.2f} GiB")
