import re, torch
m = torch.jit.load(r"D:\models\opt10\model_dir\your_model.pt", map_location="cpu").eval()
print("-- dit_step.code head --")
print(getattr(m, "dit_step").code[:1200])
hits = {"sdpa": [], "softmax": [], "conv2d": [], "linear": []}
n_mod = 0
for nm, mod in m.named_modules():
    n_mod += 1
    try:
        code = mod.code
    except Exception:
        continue
    if "scaled_dot_product" in code:
        hits["sdpa"].append(nm)
    if "softmax" in code:
        hits["softmax"].append(nm)
    if "conv2d" in code:
        hits["conv2d"].append(nm)
    if "linear" in code:
        hits["linear"].append(nm)
print("modules scanned:", n_mod)
for k, v in hits.items():
    print(f"  {k}: {len(v)} -> {v[:6]}")
