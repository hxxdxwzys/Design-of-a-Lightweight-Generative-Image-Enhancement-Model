import re, json, collections, torch
p = r"D:\models\opt10\model_dir\your_model.pt"
m = torch.jit.load(p, map_location="cpu").eval()
print("methods:", [x for x in dir(m) if x in ("encode","features","step","decode","post","forward")])
for name in ("encode", "features", "step", "decode", "post"):
    try:
        fn = getattr(m, name)
        code = fn.code if hasattr(fn, "code") else str(fn)
    except Exception as e:
        print(name, "ERR", e); continue
    ops = collections.Counter(re.findall(r"aten::(\w+)", code))
    keys = {k: ops.get(k, 0) for k in ("scaled_dot_product_attention","softmax","matmul","mm","bmm","linear",
                                       "conv2d","conv_transpose2d","upsample_bilinear2d","upsample_nearest2d",
                                       "layer_norm","group_norm","gelu","silu","mul","add","cat","pad")}
    print(f"--- {name}: lines={len(code.splitlines())} ops={sum(ops.values())}")
    print("   ", {k: v for k, v in keys.items() if v})
    if "scaled_dot_product" in code:
        print("    >>> 含 scaled_dot_product_attention")
    if "aten::softmax" in code:
        print("    >>> 含显式 softmax（可能非融合注意力）")
