import os
inner = r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战"
ws = os.path.join(inner, "workspace")
print("== inner workspace (%d entries) ==" % len(os.listdir(ws)))
for f in sorted(os.listdir(ws))[:80]:
    fp = os.path.join(ws, f)
    print(f"   {'DIR ' if os.path.isdir(fp) else 'FILE'} {f}")
print("\n== workspace/models ==")
md = os.path.join(ws, "models")
if os.path.isdir(md):
    for f in sorted(os.listdir(md)):
        fp = os.path.join(md, f)
        print("   ", f, os.path.getsize(fp) if os.path.isfile(fp) else "DIR")
print("\n== D:\\datasets ==")
dd = r"D:\datasets"
if os.path.isdir(dd):
    for f in sorted(os.listdir(dd)):
        fp = os.path.join(dd, f)
        if os.path.isdir(fp):
            n = sum(len(x) for _, _, x in os.walk(fp))
            sz = sum(os.path.getsize(os.path.join(r_, x)) for r_, _, xs in os.walk(fp) for x in xs) / 2**30
            print(f"   DIR  {f:26s} n={n:6d} {sz:8.2f} GiB")
        else:
            print(f"   FILE {f:26s} {os.path.getsize(fp)/2**20:10.1f} MiB")
else:
    print("   MISSING", dd)
