# -*- coding: utf-8 -*-
"""同步 finals 材料与文档：把旧 thr=3 的数字/哈希替换为 V4t1（thr=1）的值。"""
import io
from pathlib import Path

REPL = [
    # 权重哈希
    ("a59f3e52fd125baf", "2770215987e2b558"),
    ("2,089,751 B", "2,089,777 B"),
    ("2089751 B", "2089777 B"),
    # 感知指标（30 张测试子集，4K 原生）
    ("0.8042", "0.8102"),
    ("4.9967", "4.8555"),
    ("0.8046", "0.8102"),
    ("5.0020", "4.8555"),
    # 内置 unsharp 阈值说明
    ("内置 unsharp（60%）", "内置 unsharp（强度 60%、σ=1.2、**阈值 1/255**）"),
    ("阈值3", "阈值1"),
    ("thr=3", "thr=1"),
]

FILES = [
    Path(r"D:\models\finals_20260927\INDEX_决赛材料.md"),
    Path(r"D:\models\finals_20260927\Track2_light_diffusion_inference\使用说明.md"),
    Path(r"D:\models\finals_20260927\决赛现场操作手册.md"),
    Path(r"D:\models\finals_20260927\决赛答辩手册.md"),
]
for f in FILES:
    if not f.exists():
        print("MISSING", f); continue
    t = io.open(f, encoding="utf-8-sig").read()
    n = 0
    for a, b in REPL:
        if a in t:
            n += t.count(a); t = t.replace(a, b)
    io.open(f, "w", encoding="utf-8", newline="").write(t)
    print("updated %-64s 替换 %d 处" % (f.name, n))
