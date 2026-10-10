import os, json, glob
# 1) 待上传命名约束
d = r"D:\models\upload"
for f in os.listdir(d):
    p = os.path.join(d, f)
    if os.path.isfile(p) and f.endswith(".txt"):
        raw = open(p, "rb").read()
        for enc in ("utf-8", "gb18030"):
            try:
                print("=== %s ===" % f); print(raw.decode(enc)[:1500]); break
            except Exception:
                pass
# 2) 历史时延/指标
for j in [r"D:\tmp_timing\pkg_latency.json", r"D:\tmp_timing\four_pkgs_metrics.json", r"D:\tmp_timing\tiny_quality_diag.json"]:
    print("\n=== %s ===" % j, os.path.exists(j))
    if os.path.exists(j):
        txt = open(j, encoding="utf-8", errors="replace").read()
        print(txt[:1800])
