import re, collections, torch
m = torch.jit.load(r"D:\models\opt10\model_dir\your_model.pt", map_location="cpu").eval()
for name in ("step", "encode", "decode", "forward"):
    fn = getattr(m, name, None)
    print("="*20, name)
    try:
        print(fn.code[:900])
    except Exception as e:
        print("  code err:", e)
print("method names:", m._c._get_method_names())
# 找最重的子模块函数：遍历所有函数，统计 aten 算子
best = []
for nm, mod in m.named_modules():
    for fn_name in getattr(mod, "_c", None)._get_method_names() if hasattr(mod, "_c") else []:
        try:
            code = getattr(mod, fn_name).code
        except Exception:
            continue
        n = len(re.findall(r"aten::\w+", code))
        if n > 20:
            ops = collections.Counter(re.findall(r"aten::(\w+)", code))
            best.append((n, nm + "." + fn_name, {k: ops.get(k, 0) for k in
                        ("scaled_dot_product_attention","softmax","matmul","bmm","linear","conv2d",
                         "conv_transpose2d","upsample_bilinear2d","mul","add","cat") if ops.get(k)}))
best.sort(reverse=True)
for n, nm, ops in best[:12]:
    print(f"  {n:5d} ops  {nm}  {ops}")
