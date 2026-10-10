import os, shutil
proj = "C:\\" + "\u751f\u6210\u5f0f\u56fe\u50cf\u589e\u5f3a\u8f7b\u91cf\u6a21\u578b\u8bbe\u8ba1\u6311\u6218"
inner_eb = os.path.join(proj, "\u751f\u6210\u5f0f\u56fe\u50cf\u589e\u5f3a\u8f7b\u91cf\u6a21\u578b\u8bbe\u8ba1\u6311\u6218", "workspace", "export_baseline")
outer_ws = os.path.join(proj, "workspace")
outer_eb = os.path.join(outer_ws, "export_baseline")

moved = []
for f in sorted(os.listdir(outer_eb)):
    src, dst = os.path.join(outer_eb, f), os.path.join(inner_eb, f)
    if f.endswith(".pyc"): continue
    if os.path.exists(dst):
        print("SKIP (exists):", f); continue
    shutil.move(src, dst); moved.append(f)
print("moved:", moved)

left = [f for f in os.listdir(outer_eb) if not f.endswith(".pyc")]
print("left in outer:", left)
if not left:
    shutil.rmtree(outer_ws, ignore_errors=True)
    print("outer workspace\\ removed:", not os.path.exists(outer_ws))
print("inner export_baseline now:", len(os.listdir(inner_eb)), "entries")
for f in moved:
    print("  present:", f, os.path.getsize(os.path.join(inner_eb, f)))
