r"""refiner 形状 -> 图内时延 曲线（随机权重，只看成本；CUDA Graph 内测量）。"""
import statistics, time
import torch, torch.nn as nn, torch.nn.functional as F
dev = "cuda"

class R512(nn.Module):
    def __init__(self, ch, blocks):
        super().__init__()
        L = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks): L += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        L += [nn.Conv2d(ch, 3, 3, 1, 1)]
        self.net = nn.Sequential(*L)
    def forward(self, x): return (x + self.net(x)).clamp(-1, 1)

class R256(nn.Module):
    def __init__(self, ch, blocks):
        super().__init__()
        L = [nn.Conv2d(3, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(blocks): L += [nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True)]
        L += [nn.Conv2d(ch, 12, 3, 1, 1)]
        self.net = nn.Sequential(*L); self.ps = nn.PixelShuffle(2)
    def forward(self, x):
        return (F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False) + self.ps(self.net(x))).clamp(-1, 1)

def cost(model, size):
    m = model.to(dev).eval().half()
    x = torch.zeros(1, 3, size, size, device=dev, dtype=torch.float16)
    with torch.no_grad():
        for _ in range(3): m(x)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): m(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m(x)
        ts = []
        for _ in range(30):
            s0 = torch.cuda.Event(True); e0 = torch.cuda.Event(True)
            s0.record(); g.replay(); e0.record(); torch.cuda.synchronize()
            ts.append(s0.elapsed_time(e0))
    del m, x; torch.cuda.empty_cache()
    return statistics.median(ts)

print("== 512 域 refiner（输入 512）==")
for ch, bl in ((16,2),(16,3),(24,2),(32,1),(32,2),(48,2),(64,1),(64,2)):
    print("   ch=%2d blocks=%d : %.2f ms" % (ch, bl, cost(R512(ch, bl), 512)), flush=True)
print("== 256 域 refiner + PixelShuffle（输入 256, 输出 512）==")
for ch, bl in ((16,3),(24,3),(32,2),(32,3),(48,4),(64,3)):
    print("   ch=%2d blocks=%d : %.2f ms" % (ch, bl, cost(R256(ch, bl), 256)), flush=True)
