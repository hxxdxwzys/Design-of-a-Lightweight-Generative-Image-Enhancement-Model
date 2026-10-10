import os
roots = [r"D:\tmp_timing", r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace",
         r"D:\models", r"D:\datasets"]
pat = ("tiny", "distill")
found = []
for root in roots:
    for r_, ds, fs in os.walk(root):
        if r_.count(os.sep) - root.count(os.sep) > 3:
            ds[:] = []
            continue
        for f in fs:
            lf = f.lower()
            if lf.endswith((".py", ".pt", ".pth", ".json", ".md")) and any(p in lf for p in pat):
                fp = os.path.join(r_, f)
                found.append((os.path.getsize(fp), fp))
found.sort(reverse=True)
print("== tiny/distill artifacts ==")
for sz, fp in found[:45]:
    print(f"   {sz/2**20:9.2f} MiB  {fp}")
print("\n== distill256 / tiny_patches 结构 ==")
for d in [r"D:\datasets\distill256", r"D:\datasets\tiny_patches", r"D:\datasets\distill_hypir", r"D:\datasets\holdout_v2"]:
    if os.path.isdir(d):
        sub = sorted(os.listdir(d))
        print(f"   {d}: {len(sub)} entries -> {sub[:8]}")
        if sub and os.path.isdir(os.path.join(d, sub[0])):
            s2 = sorted(os.listdir(os.path.join(d, sub[0])))
            print(f"       {sub[0]}/: {len(s2)} -> {s2[:6]}")
