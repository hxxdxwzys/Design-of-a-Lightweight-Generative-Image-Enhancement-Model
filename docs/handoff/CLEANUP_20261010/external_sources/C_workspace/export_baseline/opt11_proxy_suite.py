r"""E2 全套代理：对一个 output_dir 计算 clipiqa@4K / niqe@4K / musiq@512crop / maniqa@512crop / lapvar@4K。
用法: python opt11_proxy_suite.py --dir DIR [--n 30] [--tag 名字]
输出: D:\models\opt11\proxy_<tag>.json
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--n", type=int, default=30)
ap.add_argument("--tag", default=""); ap.add_argument("--crop", type=int, default=512)
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
tag = a.tag or Path(a.dir).name
dev = "cuda"
ms = {k: pyiqa.create_metric(k, device=dev) for k in ("clipiqa", "niqe", "musiq", "maniqa")}
names = [f.stem for f in sorted(TEST.glob("*.jpg"))[:a.n]]
acc = {k: [] for k in ms}; laps = []
def lapvar(t):
    k = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]).view(1,1,3,3).to(t.device)
    g = torch.nn.functional.conv2d(t.mean(1, keepdim=True), k, padding=1)
    return float(g.var()*255.0*255.0)
t0 = time.time(); used = 0
for nm in names:
    p = Path(a.dir) / ("%s.jpg" % nm)
    if not p.exists():
        continue
    im = Image.open(p).convert("RGB")
    arr = np.asarray(im)
    h, w = arr.shape[:2]
    c = a.crop
    y0, x0 = (h-c)//2, (w-c)//2
    arrc = arr[y0:y0+c, x0:x0+c]
    t4 = torch.from_numpy(arr.copy()).permute(2,0,1).unsqueeze(0).to(dev).float().div_(255)
    tc = torch.from_numpy(arrc.copy()).permute(2,0,1).unsqueeze(0).to(dev).float().div_(255)
    with torch.no_grad():
        acc["clipiqa"].append(float(ms["clipiqa"](t4).item()))
        acc["niqe"].append(float(ms["niqe"](t4).item()))
        acc["musiq"].append(float(ms["musiq"](tc).item()))
        acc["maniqa"].append(float(ms["maniqa"](tc).item()))
        laps.append(lapvar(t4))
    used += 1
    del t4, tc
out = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
out["lapvar"] = round(float(np.mean(laps)), 1)
out["n"] = used; out["dir"] = a.dir; out["tag"] = tag
Path(r"D:\models\opt11").mkdir(parents=True, exist_ok=True)
Path(r"D:\models\opt11\proxy_%s.json" % tag).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("PROXY %s -> %s" % (tag, json.dumps(out, ensure_ascii=False)), flush=True)
print("  (%.1f s, %d 张)" % (time.time()-t0, used), flush=True)
print("PROXY_SUITE_DONE", flush=True)
