r"""裁剪尺度 clipiqa（512 中心裁剪）逐图测量，用于检查"裁剪 vs 全图"尺度一致性（E15 教训）。"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"): os.environ.pop(k, None)
import numpy as np, torch, pyiqa
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--tag", default=""); ap.add_argument("--n", type=int, default=30); ap.add_argument("--crop", type=int, default=512)
a = ap.parse_args()
tag = a.tag or Path(a.dir).name
dev = "cuda"; m = pyiqa.create_metric("clipiqa", device=dev)
names = [p.stem for p in sorted(Path(a.dir).glob("*.jpg"))][:a.n]
vals = []
for nm in names:
    p = Path(a.dir) / (nm + ".jpg")
    arr = np.asarray(Image.open(p).convert("RGB")); h, w = arr.shape[:2]; c = min(a.crop, h, w)
    y0, x0 = (h-c)//2, (w-c)//2
    t = torch.from_numpy(arr[y0:y0+c, x0:x0+c].copy()).permute(2,0,1).unsqueeze(0).to(dev).float().div_(255)
    with torch.no_grad(): vals.append(float(m(t).item()))
out = {"tag": tag, "dir": a.dir, "clipiqa_crop_mean": round(float(np.mean(vals)), 4), "n": len(vals), "crop": a.crop}
Path(r"D:\models\opt11\cropclip_%s.json" % tag).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("CROPCLIP " + json.dumps(out, ensure_ascii=False), flush=True)
print("CROPCLIP_DONE", flush=True)
