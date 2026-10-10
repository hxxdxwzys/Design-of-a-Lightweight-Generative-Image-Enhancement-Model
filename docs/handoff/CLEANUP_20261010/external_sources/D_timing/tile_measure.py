# -*- coding: utf-8 -*-
"""T 系列回归的度量：对 tile_<tag>/ 目录算【原生 4K 全图】clipiqa / niqe / lapvar。
（musiq@4K = 42.8 s/张，太贵，放到第二层；maniqa@4K 只对 512 裁切敏感，另列。）
用法: python tile_measure.py <tag> [<tag> ...]
"""
import os, sys, json, time
os.environ["HF_HUB_OFFLINE"] = "1"
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = r"D:\models\opt11"
dev = "cuda"
tags = sys.argv[1:]
assert tags

ms = {k: pyiqa.create_metric(k, device=dev) for k in ("clipiqa", "niqe")}


def lapvar(t4):
    """灰度 Laplacian 方差 —— 与 D:\\tmp_timing\\measure_dir.py 同定义（保证与历史读数可比）。"""
    g = (0.299 * t4[:, 0] + 0.587 * t4[:, 1] + 0.114 * t4[:, 2]).unsqueeze(1) * 255.0
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3).to(t4.device)
    lap = torch.nn.functional.conv2d(g, k, padding=1)
    return float(lap.var())


rows = {}
for tag in tags:
    d = os.path.join(OUT, "tile_%s" % tag)
    if not os.path.isdir(d):
        print("MISSING", d); continue
    files = sorted(os.listdir(d), key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))
    acc = {k: [] for k in ms}; laps = []; per = []
    t0 = time.time()
    for f in files:
        im = Image.open(os.path.join(d, f)).convert("RGB")
        arr = np.asarray(im)
        t4 = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).to(dev).float().div_(255)
        with torch.no_grad():
            c = float(ms["clipiqa"](t4).item()); n = float(ms["niqe"](t4).item()); l = lapvar(t4)
        acc["clipiqa"].append(c); acc["niqe"].append(n); laps.append(l)
        per.append({"case": f, "clipiqa": round(c, 4), "niqe": round(n, 4), "lapvar": round(l, 1)})
        del t4
    rec = {"tag": tag, "n": len(files), "dir": d,
           "clipiqa": round(float(np.mean(acc["clipiqa"])), 4),
           "niqe": round(float(np.mean(acc["niqe"])), 4),
           "lapvar": round(float(np.mean(laps)), 1),
           "clipiqa_std": round(float(np.std(acc["clipiqa"])), 4),
           "niqe_std": round(float(np.std(acc["niqe"])), 4),
           "per_image": per, "sec": round(time.time() - t0, 1)}
    rows[tag] = rec
    with open(os.path.join(OUT, "tilem_%s.json" % tag), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print("MEASURE %-8s n=%d  clipiqa=%.4f(±%.4f)  niqe=%.4f(±%.4f)  lapvar=%.1f  [%.0f s]"
          % (tag, rec["n"], rec["clipiqa"], rec["clipiqa_std"], rec["niqe"], rec["niqe_std"], rec["lapvar"], rec["sec"]),
          flush=True)

# 配对差（相对第一个 tag）
if len(rows) > 1:
    base_tag = tags[0]
    if base_tag in rows:
        b = {p["case"]: p for p in rows[base_tag]["per_image"]}
        print("\n配对逐图差（相对 %s）:" % base_tag)
        print("%-8s %10s %10s %10s %8s" % ("tag", "d_clipiqa", "d_niqe", "d_lapvar", "胜/负"))
        for tag in tags[1:]:
            if tag not in rows: continue
            dc, dn, dl, win, lose = [], [], [], 0, 0
            for p in rows[tag]["per_image"]:
                if p["case"] not in b: continue
                q = b[p["case"]]
                dc.append(p["clipiqa"] - q["clipiqa"]); dn.append(p["niqe"] - q["niqe"]); dl.append(p["lapvar"] - q["lapvar"])
                # 判胜：clipiqa 更高 且 niqe 更低
                if p["clipiqa"] > q["clipiqa"] and p["niqe"] < q["niqe"]: win += 1
                elif p["clipiqa"] < q["clipiqa"] and p["niqe"] > q["niqe"]: lose += 1
            print("%-8s %+10.4f %+10.4f %+10.1f %4d/%d" % (tag, np.mean(dc), np.mean(dn), np.mean(dl), win, lose))
print("TILE_MEASURE_DONE")
