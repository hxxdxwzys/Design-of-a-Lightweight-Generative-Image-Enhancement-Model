
import os, sys
MD = os.path.abspath(sys.argv[1])
sys.path.insert(0, MD)          # 官方 harness 的标准做法
import runner                    # noqa: F401
import timm, einops
def outside(m):
    f = os.path.abspath(getattr(m, "__file__", "") or "")
    return not f.startswith(MD + os.sep)
ok = outside(timm) and outside(einops)
try:
    from timm.models.vision_transformer import Block   # noqa: F401
    block_ok = True
except Exception:
    block_ok = False
print("PROBE pass=%s timm_outside=%s einops_outside=%s block_ok=%s" % (ok and block_ok, outside(timm), outside(einops), block_ok))
print("PROBE timm=", os.path.abspath(getattr(timm, "__file__", "")))
print("PROBE einops=", os.path.abspath(getattr(einops, "__file__", "")))
sys.exit(0 if (ok and block_ok) else 1)
