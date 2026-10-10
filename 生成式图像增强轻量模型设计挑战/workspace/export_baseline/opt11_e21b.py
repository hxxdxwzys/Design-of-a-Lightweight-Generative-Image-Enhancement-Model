r"""E21b: 直接在 arch 层测 clipiqa 可微性 + 确认内部缩放与 no_grad 位置。"""
import io, os, re, sys
os.environ["HF_HUB_OFFLINE"] = "1"
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"): os.environ.pop(k, None)
import torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p1 = r"D:\Python\Lib\site-packages\pyiqa\archs\clipiqa_arch.py"
p2 = r"D:\Python\Lib\site-packages\pyiqa\archs\clip_model.py"
for p in (p1, p2):
    s = io.open(p, encoding="utf-8").read()
    print("==", os.path.basename(p), "no_grad 出现次数:", s.count("no_grad"), " reshape/interp:", s.count("interpolate"))
    for m in re.finditer(r".*no_grad.*", s):
        print("    ", m.group(0).strip()[:110])
print()
from pyiqa.archs.clipiqa_arch import CLIPIQA
m = CLIPIQA(model_type="clipiqa", backbone="RN50")
m.eval()
x = torch.rand(1, 3, 224, 224, requires_grad=True)
y = m(x)
print("arch 直接调用: score=%.5f requires_grad=%s shape=%s" % (float(y), y.requires_grad, tuple(y.shape)))
if y.requires_grad:
    y.sum().backward()
    print("  ✓ 可微 |g|mean=%.3e max=%.3e" % (x.grad.abs().mean(), x.grad.abs().max()))
x2 = torch.rand(1, 3, 512, 512, requires_grad=True)
y2 = m(x2)
print("512 输入: score=%.5f requires_grad=%s" % (float(y2), y2.requires_grad))
if y2.requires_grad:
    y2.sum().backward(); print("  ✓ 512 可微 |g|mean=%.3e" % x2.grad.abs().mean())
print("E21B_DONE")
