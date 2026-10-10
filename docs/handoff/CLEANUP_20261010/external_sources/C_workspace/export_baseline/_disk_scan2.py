import os
def dsize(p):
    t = 0
    for r_, _, fs in os.walk(p):
        for f in fs:
            try: t += os.path.getsize(os.path.join(r_, f))
            except Exception: pass
    return t / 2**30
def top(p, n=12):
    rows = []
    if not os.path.isdir(p): return rows
    for e in sorted(os.listdir(p)):
        fp = os.path.join(p, e)
        rows.append((dsize(fp) if os.path.isdir(fp) else os.path.getsize(fp) / 2**30,
                     ("DIR  " if os.path.isdir(fp) else "FILE ") + e))
    return sorted(rows, reverse=True)[:n]
for d in [r"D:\models\newsrc", r"D:\tmp_timing", r"D:\models\opt10", r"D:\models\opt01", r"D:\datasets"]:
    print("==", d)
    for s, n in top(d):
        print(f"   {s:8.2f} GiB  {n}")
    print()
