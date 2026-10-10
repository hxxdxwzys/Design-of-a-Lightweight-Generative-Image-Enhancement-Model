import json
d = json.load(open(r"D:\models\opt10\ledger_opt10.json", encoding="utf-8"))
print("limits:", d["limits"]); print("prior:", d.get("prior_batch")); print("used:", d["used"])
print("n events:", len(d.get("events") or []), "attempts:", len(d.get("attempts") or []))
