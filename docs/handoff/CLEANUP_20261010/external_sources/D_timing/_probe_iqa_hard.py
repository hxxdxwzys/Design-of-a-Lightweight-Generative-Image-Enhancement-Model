
import os, sys
MD = os.path.abspath(sys.argv[1])
sys.path.insert(0, MD)          # ① 官方 harness: 插 model_dir 以便 import runner
import runner                    # ② 我们的 runner (内部会做清理)
sys.path.insert(0, MD)          # ③ 关键: harness 之后再插一次 (对抗情形)
import timm, einops
def outside(m):
    f = os.path.abspath(getattr(m, "__file__", "") or "")
    return not f.startswith(MD + os.sep)
try:
    from timm.models.vision_transformer import Block   # noqa: F401
    block_ok = True
except Exception as e:
    block_ok = False
ok = outside(timm) and outside(einops) and block_ok
print("HARD pass=%s timm_outside=%s einops_outside=%s block_ok=%s" % (ok, outside(timm), outside(einops), block_ok))
print("HARD timm=", os.path.abspath(getattr(timm, "__file__", "")))
print("HARD einops=", os.path.abspath(getattr(einops, "__file__", "")))
sys.exit(0 if ok else 1)
