import json, io, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = r"D:\tmp_timing\score_ledger.json"
d = json.load(io.open(p, encoding="utf-8"))
print("type:", type(d).__name__)
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from walk(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from walk(v, path + "[%d]" % i)
    else:
        yield path, o
targets = ["68.6681", "70.1021", "69.4540", "66.2813", "68.6300", "68.67", "70.10", "69.45", "66.28", "68.63"]
hits = []
for k, v in walk(d):
    s = str(v)
    for t in targets:
        if t in s: hits.append((k, s[:120]))
print("hits:", len(hits))
for k, s in hits[:40]: print("  ", k, "=", s)