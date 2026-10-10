import os, time, subprocess
proj = "C:\\" + "\u751f\u6210\u5f0f\u56fe\u50cf\u589e\u5f3a\u8f7b\u91cf\u6a21\u578b\u8bbe\u8ba1\u6311\u6218"
inner = os.path.join(proj, "\u751f\u6210\u5f0f\u56fe\u50cf\u589e\u5f3a\u8f7b\u91cf\u6a21\u578b\u8bbe\u8ba1\u6311\u6218")

print("inner is symlink:", os.path.islink(inner))
try:
    st = os.stat(inner)
    print("inner attrs:", hex(getattr(st, "st_file_attributes", 0)), "reparse:", bool(getattr(st, "st_file_attributes", 0) & 0x400))
except Exception as e:
    print("stat err", e)

eb = os.path.join(inner, "workspace", "export_baseline")
fs = [(os.path.getmtime(os.path.join(eb, f)), f) for f in os.listdir(eb)]
fs.sort(reverse=True)
print(f"\nINNER export_baseline: {len(fs)} entries, newest 30:")
for t, f in fs[:30]:
    print(f"   {time.strftime('%m-%d %H:%M', time.localtime(t))}  {f}")
today = [f for t, f in fs if time.strftime('%m-%d', time.localtime(t)) == '09-19']
print("modified today (09-19):", len(today), today[:20])

print("\nOUTER workspace tree:")
for root, dirs, files in os.walk(os.path.join(proj, "workspace")):
    rel = os.path.relpath(root, proj)
    print(f"   {rel}: {len(files)} files")
    if len(files) > 40: break

print("\nouter .gitignore:")
gi = os.path.join(proj, ".gitignore")
print(open(gi, encoding="utf-8", errors="replace").read()[:600] if os.path.exists(gi) else "MISSING")

r = subprocess.run(["git", "status", "--short"], cwd=proj, capture_output=True, text=True, encoding="utf-8", errors="replace")
lines = [l for l in (r.stdout or "").splitlines()]
print(f"\ngit status: {len(lines)} lines, first 12:")
print("\n".join(lines[:12]))
r2 = subprocess.run(["git", "log", "--oneline", "-3"], cwd=proj, capture_output=True, text=True, encoding="utf-8", errors="replace")
print("git log:\n" + (r2.stdout or "").strip()[:400])
