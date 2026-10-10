import os
from PIL import Image
import numpy as np
d = r"D:\datasets\distill256"
for sub in ("lq", "tgt"):
    p = os.path.join(d, sub)
    fs = sorted(os.listdir(p))
    print(sub, len(fs), "entries; first:", fs[:3])
    for f in fs[:3]:
        im = Image.open(os.path.join(p, f))
        print("   ", f, im.size, im.mode)
    # 检查同名对应
names_lq = set(os.listdir(os.path.join(d, "lq")))
names_tgt = set(os.listdir(os.path.join(d, "tgt")))
print("同名交集:", len(names_lq & names_tgt), "| lq 独有:", len(names_lq - names_tgt), "| tgt 独有:", len(names_tgt - names_lq))
# tgt 与 M24 的关系？看一张 tgt 的统计
im = np.asarray(Image.open(os.path.join(d, "tgt", sorted(names_tgt)[0])).convert("RGB"), dtype=np.float32)
print("tgt 统计: mean %.1f std %.1f" % (im.mean(), im.std()))
