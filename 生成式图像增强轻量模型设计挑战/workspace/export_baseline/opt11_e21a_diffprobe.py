r"""E21a: clipiqa 对输入可微性探针（纯 CPU，避免与 E19 抢 GPU）。"""
import os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"): os.environ.pop(k, None)
import torch, pyiqa
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("building clipiqa on cpu ...", flush=True)
t0 = time.time()
m = pyiqa.create_metric("clipiqa", device="cpu")
print("built %.1fs" % (time.time()-t0), flush=True)
x = torch.rand(1, 3, 224, 224, requires_grad=True)
y = m(x)
print("score =", float(y), " requires_grad =", y.requires_grad, " shape =", tuple(y.shape), flush=True)
if y.requires_grad:
    y.sum().backward()
    g = x.grad
    print("grad ok: |g| mean=%.3e max=%.3e  nonzero=%.1f%%" % (g.abs().mean(), g.abs().max(), 100*(g.abs() > 0).float().mean()), flush=True)
# 也测 512 输入（我们裁剪尺度）是否被内部缩放
x2 = torch.rand(1, 3, 512, 512, requires_grad=True)
t1 = time.time(); y2 = m(x2); dt = time.time()-t1
print("512 输入 score=%.4f  前向 %.2fs  requires_grad=%s" % (float(y2), dt, y2.requires_grad), flush=True)
if y2.requires_grad:
    y2.sum().backward(); print("512 反传 ok |g|mean=%.3e" % x2.grad.abs().mean(), flush=True)
print("E21A_DONE", flush=True)
