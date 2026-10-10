import os, glob
proj = "C:\\" + "\u751f\u6210\u5f0f\u56fe\u50cf\u589e\u5f3a\u8f7b\u91cf\u6a21\u578b\u8bbe\u8ba1\u6311\u6218"
for d in [proj, os.path.join(proj, "workspace", "export_baseline"), r"C:\workspace"]:
    print("--", d, "exists:", os.path.isdir(d))
    if os.path.isdir(d):
        fs = sorted(os.listdir(d))
        print("   entries:", len(fs), fs[:14])
eb = os.path.join(proj, "workspace", "export_baseline")
for f in ["score01r_profile.py", "score01r_bench.py", "chain.py", "delivery_runner.py", "pixel_io.py", "server_probe.py"]:
    p = os.path.join(eb, f)
    print(f"   {f:26s} {'OK ' + str(os.path.getsize(p)) if os.path.exists(p) else 'MISSING'}")
print("cwd:", os.getcwd())
