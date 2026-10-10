import os, time
proj = "C:\\" + "\u751f\u6210\u5f0f\u56fe\u50cf\u589e\u5f3a\u8f7b\u91cf\u6a21\u578b\u8bbe\u8ba1\u6311\u6218"
inner = os.path.join(proj, "\u751f\u6210\u5f0f\u56fe\u50cf\u589e\u5f3a\u8f7b\u91cf\u6a21\u578b\u8bbe\u8ba1\u6311\u6218")

def probe(root, label):
    print(f"=== {label}: {root} ===")
    if not os.path.isdir(root):
        print("  MISSING"); return
    for sub in ["docs", "docs/handoff", "docs/reference", "docs/archive", "workspace", "workspace/export_baseline", ".git"]:
        p = os.path.join(root, *sub.split("/"))
        if os.path.isdir(p):
            n = len(os.listdir(p))
            newest = ""
            try:
                fs = [(os.path.getmtime(os.path.join(p, f)), f) for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
                if fs:
                    t, f = max(fs)
                    newest = f" newest={f} @{time.strftime('%m-%d %H:%M', time.localtime(t))}"
            except Exception as e:
                newest = f" err={e}"
            print(f"  {sub:26s} exists  n={n}{newest}")
        else:
            print(f"  {sub:26s} MISSING")
    for f in ["AGENTS.md", "README.md", "docs/progress.md", "docs/handoff/INDEX.md"]:
        p = os.path.join(root, *f.split("/"))
        print(f"  {f:26s} {'OK ' + str(os.path.getsize(p)) + ' @' + time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(p))) if os.path.exists(p) else 'MISSING'}")

probe(proj, "OUTER")
probe(inner, "INNER")
print("=== top-level of outer ===")
for e in sorted(os.listdir(proj)):
    p = os.path.join(proj, e)
    print("  ", "DIR " if os.path.isdir(p) else "FILE", e)
