import os, zipfile
def dsize(p):
    t = 0
    for r_, _, fs in os.walk(p):
        for f in fs:
            try: t += os.path.getsize(os.path.join(r_, f))
            except Exception: pass
    return t / 2**30
print("== D:\\ 顶层占用 ==")
rows = []
for e in sorted(os.listdir("D:\\")):
    p = os.path.join("D:\\", e)
    if os.path.isdir(p):
        rows.append((dsize(p), "DIR  " + e))
    elif os.path.isfile(p):
        rows.append((os.path.getsize(p) / 2**30, "FILE " + e))
for s, n in sorted(rows, reverse=True)[:14]:
    print(f"   {s:8.2f} GiB  {n}")
print("\n== D:\\models 子目录 ==")
rows = []
for e in sorted(os.listdir(r"D:\models")):
    p = os.path.join(r"D:\models", e)
    rows.append((dsize(p) if os.path.isdir(p) else os.path.getsize(p) / 2**30, ("DIR  " if os.path.isdir(p) else "FILE ") + e))
for s, n in sorted(rows, reverse=True)[:12]:
    print(f"   {s:8.2f} GiB  {n}")
print("\n== div2k.zip 与 div2k/ 一致性 ==")
zp = r"D:\datasets\div2k.zip"; dd = r"D:\datasets\div2k"
if os.path.exists(zp) and os.path.isdir(dd):
    with zipfile.ZipFile(zp) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    have = 0
    miss = []
    for n in names:
        cand = os.path.join(dd, os.path.basename(n))
        if os.path.exists(cand): have += 1
        else: miss.append(n)
    print(f"   zip 内文件 {len(names)}，已解出同名 {have}，缺 {len(miss)} -> {miss[:5]}")
print("\n== _harness_zip_sim ==")
h = r"D:\models\_harness_zip_sim"
print("   ", os.path.isdir(h), round(dsize(h), 3), "GiB" if os.path.isdir(h) else "")
