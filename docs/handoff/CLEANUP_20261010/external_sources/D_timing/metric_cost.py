# -*- coding: utf-8 -*-
"""测量各代理在 4K 原图 / 512 裁切上的单张耗时，决定回归扫描的预算。"""
import os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TEST = r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集"
dev = "cuda"
p = os.path.join(TEST, "case3.jpg")
im = Image.open(p).convert("RGB")
arr = np.asarray(im)
print("size", arr.shape)
t4 = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
c = 512
y0, x0 = (arr.shape[0] - c) // 2, (arr.shape[1] - c) // 2
tc = torch.from_numpy(arr[y0:y0+c, x0:x0+c].copy()).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)

for name in ("clipiqa", "niqe", "musiq", "maniqa"):
    try:
        m = pyiqa.create_metric(name, device=dev)
    except Exception as e:
        print("%-9s FAILED to create: %s" % (name, e)); continue
    for label, t in (("4K", t4), ("512crop", tc)):
        try:
            with torch.no_grad():
                m(t)
                torch.cuda.synchronize()
                t0 = time.time()
                v = m(t).item()
                torch.cuda.synchronize()
                dt = time.time() - t0
            print("%-9s %-8s %.3f   %.2f s/张" % (name, label, v, dt), flush=True)
        except Exception as e:
            print("%-9s %-8s ERROR: %s" % (name, label, str(e)[:120]), flush=True)
