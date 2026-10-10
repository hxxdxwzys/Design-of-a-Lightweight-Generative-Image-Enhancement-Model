r"""验证 pyiqa 的 musiq/maniqa 是否可微（能否作训练损失），并测其前向成本。"""
import os, time
os.environ["HF_HUB_OFFLINE"] = "1"
import torch, pyiqa
dev = "cuda"
for name in ("musiq", "maniqa", "clipiqa"):
    try:
        m = pyiqa.create_metric(name, device=dev)
        x = torch.rand(2, 3, 256, 256, device=dev, requires_grad=True)
        t0 = time.time(); s = m(x); fwd = time.time()-t0
        t0 = time.time(); s.sum().backward(); bwd = time.time()-t0
        g = x.grad
        print("%-8s 前向 %.3f s  反向 %.3f s  可微=%s  grad_norm=%.3e  分数范围=%s" %
              (name, fwd, bwd, g is not None, float(g.abs().mean()) if g is not None else -1, [round(float(v),3) for v in s]))
    except Exception as e:
        print("%-8s 失败: %s" % (name, str(e)[:100]))
# 训练时若用 256 crop + bs 4 的成本
m = pyiqa.create_metric("musiq", device=dev)
x = torch.rand(4, 3, 256, 256, device=dev, requires_grad=True)
t0=time.time()
for _ in range(5):
    s = m(x); loss = -s.mean(); loss.backward()
print("musiq bs=4 256crop: 单步(前+反) %.1f ms" % ((time.time()-t0)/5*1000))
