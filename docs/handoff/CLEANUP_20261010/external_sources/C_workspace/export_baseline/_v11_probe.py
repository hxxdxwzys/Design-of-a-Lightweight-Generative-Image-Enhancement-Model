import os, json, glob
d = r"D:\models\pkg_TinyV11b"
for r_, ds, fs in os.walk(d):
    lvl = r_.count(os.sep) - d.count(os.sep)
    if lvl <= 1:
        print(os.path.relpath(r_, d), "dirs:", ds[:5], "files:", len(fs), fs[:4])
p = os.path.join(d, "model_dir")
if os.path.isdir(p):
    for f in sorted(os.listdir(p)):
        fp = os.path.join(p, f); print("   model_dir:", f, os.path.getsize(fp) if os.path.isfile(fp) else "DIR")
od = os.path.join(d, "output_dir")
if os.path.isdir(od):
    fs = sorted(os.listdir(od)); print("   output_dir:", len(fs), fs[:3], os.path.getsize(os.path.join(od, fs[0])) if fs else "")

print("\n== 训练检查点（v11 / tiny 学生）==")
pats = [r"D:\models\**\v11*.pt", r"D:\models\**\tiny*.pt", r"D:\tmp_timing\**\v1*.pt", r"D:\tmp_timing\**\*.pt"]
seen = set()
for pat in pats:
    for f in glob.glob(pat, recursive=True):
        if f in seen: continue
        seen.add(f)
        try: print("   %.1f MiB  %s" % (os.path.getsize(f)/2**20, f))
        except Exception: pass
print("\n== 训练/评测/渲染脚本 ==")
for pat in ["chain_20260915*.py", "train_tiny_gan.py", "train_tiny*.py", "render_tiny.py", "render_subset.py", "measure_dir.py", "measure_test30*.py", "diag_tiny_quality.py", "eval_tiny.py"]:
    hits = glob.glob(r"D:\tmp_timing\\" + pat)
    if hits: print("   ", hits)
