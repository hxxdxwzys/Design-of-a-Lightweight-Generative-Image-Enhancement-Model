import re, collections, torch
m = torch.jit.load(r"D:\models\opt10\model_dir\your_model.pt", map_location="cpu").eval()
for nm in ("dit_step", "vae_enc_512", "vae_enc_1024", "dec_64", "dec_128", "feature_extractor"):
    sub = getattr(m, nm, None)
    if sub is None:
        print("-- missing:", nm); continue
    try:
        code = sub.code
    except Exception as e:
        print("--", nm, "no code:", e); continue
    ops = collections.Counter(re.findall(r"aten::(\w+)", code))
    sel = {k: ops[k] for k in ("scaled_dot_product_attention","softmax","matmul","mm","bmm","linear","conv2d",
                               "conv_transpose2d","upsample_bilinear2d","upsample_nearest2d","mul","add","cat",
                               "split","chunk","contiguous","permute","view","layer_norm","group_norm","gelu","silu") if ops.get(k)}
    print(f"== {nm}: lines={len(code.splitlines())} total_ops={sum(ops.values())}")
    print("   ", sel)
    print("    sdpa:", "scaled_dot_product_attention" in code, "| softmax:", "aten::softmax" in code)
