# -*- coding: utf-8 -*-
"""把 S1 训练/导出链修正为正确形式：
  学生输出@S -> (双三次↓S/2) -> U↑S -> refiner@S
并且 U 采用【残差学习】：U(x) = bicubic(x) + ps(c2(...))，zero-init 起点 = bicubic。
诊断（已实测）：当前 U 与 bicubic 偏差 mean 0.152/max 1.24 => U 受 11K refiner 容量限制无法学到恒等，
所以"U 在 S/2 域"不可行；必须让 U 的工作域与 refiner 一致（都在 S 域，U 只做 S/2->S 的残差细化）。
"""
from pathlib import Path

# ---------- 1) 训练器：加 --size ----------
p = Path(r"D:\tmp_timing\s1_train.py")
t = p.read_text(encoding="utf-8-sig")

a1 = 'ap.add_argument("--init_up"'
assert a1 in t
t = t.replace(a1, 'ap.add_argument("--size", type=int, default=512, help="学生输出域尺寸(=refiner 工作域)")\n' + a1, 1)

# 断言链的域：in 是 S 域；down 到 S/2；U 回 S；refiner 在 S
old_comment = "# 冻结的学生（只用来产 512 域底图）"
if old_comment in t:
    t = t.replace(old_comment, "# 冻结的学生（只用来产 S 域底图；S 由 --size 指定）", 1)

p.write_text(t, encoding="utf-8")

# ---------- 2) 导出器：同域 ----------
q = Path(r"D:\tmp_timing\s1_export.py")
s = q.read_text(encoding="utf-8-sig")
s = s.replace('ap.add_argument("--thr"', 'ap.add_argument("--size", type=int, default=512)\nap.add_argument("--thr"', 1)
s = s.replace("""        xs = F.interpolate(x, size=(256, 256), mode="bicubic", align_corners=False)
        pred = self.unet(xs, self.tt).sample
        y = (xs - self.sq_1a.to(x.dtype) * pred) / self.sq_a.to(x.dtype)
        y512 = F.interpolate(y, size=(512, 512), mode="bicubic", align_corners=False)   # 512 域底图
        return self._unsharp(self.G(self.U(y512)))""",
"""        S = int(self.size)
        xs = F.interpolate(x, size=(S, S), mode="bicubic", align_corners=False)
        pred = self.unet(xs, self.tt).sample
        y = (xs - self.sq_1a.to(x.dtype) * pred) / self.sq_a.to(x.dtype)   # 学生输出，S 域
        return self._unsharp(self.G(self.U(y)))""", 1)
s = s.replace("""    def __init__(self, unet, U, G, sq_a, sq_1a, pct, sigma, thr):
        super().__init__()
        self.unet = unet; self.U = U; self.G = G
        self.pct = float(pct); self.sigma = float(sigma); self.thr = float(thr)""",
"""    def __init__(self, unet, U, G, sq_a, sq_1a, pct, sigma, thr, size=512):
        super().__init__()
        self.unet = unet; self.U = U; self.G = G
        self.pct = float(pct); self.sigma = float(sigma); self.thr = float(thr); self.size = int(size)""", 1)
s = s.replace("M = Combined(unet, U, G, sq_a, sq_1a, a.pct, a.sigma, a.thr).to(dev).eval().half()",
              "M = Combined(unet, U, G, sq_a, sq_1a, a.pct, a.sigma, a.thr, a.size).to(dev).eval().half()", 1)
q.write_text(s, encoding="utf-8")

import io
for f in (p, q):
    compile(io.open(f, encoding="utf-8").read(), f, "exec")
    print("compile OK:", f)
