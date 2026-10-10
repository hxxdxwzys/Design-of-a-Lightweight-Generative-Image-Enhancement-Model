import json
from collections import Counter
sr = json.load(open(r"D:/models/sweep_test_out/sweep_results.json", encoding="utf-8"))
for r in sr["results"]:
    if r["name"].startswith("vosr_lora"):
        print(f'{r["name"]}: {r["test_clipiqa"]}')
cm = json.load(open(r"D:/tmp_timing/choice_maps.json", encoding="utf-8"))
for m in ("M0.000", "M0.010"):
    mm = cm["maps"][m]
    print(f'{m}: mean={mm["mean"]} swaps={mm["n_swap"]}')
print(Counter(cm["maps"]["M0.010"]["choice"].values()).most_common(8))
