import os, time, subprocess
proj = "C:\\" + "\u751f\u6210\u5f0f\u56fe\u50cf\u589e\u5f3a\u8f7b\u91cf\u6a21\u578b\u8bbe\u8ba1\u6311\u6218"
inner = os.path.join(proj, "\u751f\u6210\u5f0f\u56fe\u50cf\u589e\u5f3a\u8f7b\u91cf\u6a21\u578b\u8bbe\u8ba1\u6311\u6218")

print("=== git status untracked/workspace lines ===")
r = subprocess.run(["git", "status", "--short"], cwd=proj, capture_output=True, text=True, encoding="utf-8", errors="replace")
for l in (r.stdout or "").splitlines():
    if l.startswith("??") or "workspace" in l:
        print("  ", l[:150])

print("\n=== inner _s31.md (head) ===")
p = os.path.join(inner, "workspace", "export_baseline", "_s31.md")
if os.path.exists(p):
    print("size", os.path.getsize(p))
    print(open(p, encoding="utf-8", errors="replace").read()[:900])

print("\n=== artifacts ===")
arts = [
 r"C:\models\exports\DELIV02_20260918_073810\Track2_A2_fp16_v1\u2014\u201442.2145.zip".replace("\u2014","\u2014"),
 r"D:\models\opt01\Track2_OPT01_fp16full_v1.zip",
 r"D:\models\opt10\Track2_OPT10_fp16full_unsharp_v1.zip",
 r"D:\models\opt10\ledger_opt10.json",
 r"D:\models\opt10\evidence",
 r"D:\models\opt01\evidence",
 r"D:\models\iqa_screen",
 r"D:\models\finals\preset\ckpts\VOSR_0.5B_os",
 r"D:\models\vosr_lora\v7\lora_s5000.pt",
]
for a in arts:
    if os.path.isdir(a):
        n = sum(len(f) for _, _, f in os.walk(a))
        print(f"  DIR  {a}  ({n} files)")
    elif os.path.exists(a):
        print(f"  FILE {a}  {os.path.getsize(a):,} B")
    else:
        print(f"  MISS {a}")

print("\n=== outer docs/progress.md tail ===")
pp = os.path.join(proj, "docs", "progress.md")
txt = open(pp, encoding="utf-8", errors="replace").read()
print("size", len(txt), "chars; last 700 chars:")
print(txt[-700:])
