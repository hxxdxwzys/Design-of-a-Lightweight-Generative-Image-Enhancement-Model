r"""打印 v21 学生的架构（参数形状）与粗略 FLOPs 分布，判断"削哪里最划算"。"""
import sys
import torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CK = r"D:\models\tiny_v1\v21_s21000.pt"
d = torch.load(CK, map_location="cpu", weights_only=False)
sd = d.get("net", d) if isinstance(d, dict) else d
print("顶层键:", list(d.keys())[:8] if isinstance(d, dict) else type(d))
items = list(sd.items())
print("参数张量数:", len(items))
tot = 0
for k, v in items:
    if hasattr(v, "shape"):
        n = v.numel(); tot += n
        if v.ndim == 4 and min(v.shape[-2:]) > 1:
            print("  %-46s %s  %d" % (k, tuple(v.shape), n))
print("Conv 权重参数合计: %.3f M" % (tot/1e6))
cfg = {k: v for k, v in (d.items() if isinstance(d, dict) else []) if k != "net"}
print("配置项:", {k: (v if not hasattr(v, "shape") else tuple(v.shape)) for k, v in cfg.items()})
