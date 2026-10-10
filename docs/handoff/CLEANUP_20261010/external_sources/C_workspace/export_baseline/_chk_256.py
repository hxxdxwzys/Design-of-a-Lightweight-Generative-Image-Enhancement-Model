import io, json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(io.open(r"D:\tmp_timing\score_ledger.json", encoding="utf-8"))
s = json.dumps(d, ensure_ascii=False)
for k in ("256.3", "275.9", "256", "275"):
    print(k, "出现次数:", s.count(k))
print("ledger 顶层键:", list(d.keys())[:15] if isinstance(d, dict) else type(d))