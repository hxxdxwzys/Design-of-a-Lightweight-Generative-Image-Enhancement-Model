import io, json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for p in (r"D:\tmp_timing\speed_breakdown_shipped.json", r"D:\tmp_timing\speed_breakdown_512.json"):
    try:
        d = json.load(io.open(p, encoding="utf-8"))
    except Exception as e:
        print(p, "ERR", e); continue
    print("==", p)
    print(json.dumps(d, ensure_ascii=False, indent=1)[:1800])