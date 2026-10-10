r"""E10b：4K 口径 musiq/maniqa 补测（每目录前 n 张三张），检验 crop 排序与 4K 排序是否一致。
用法: python opt11_e10b_4k.py <dir_or_zip> <tag> [n]
"""
import io, os, sys, time, zipfile
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch, pyiqa
from PIL import Image
from pathlib import Path
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
src = sys.argv[1]; tag = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
dev = "cuda"
ms = {k: pyiqa.create_metric(k, device=dev) for k in ("musiq", "maniqa")}
names = [f.stem for f in sorted(TEST.glob("*.jpg"))[:n]]
res = {k: [] for k in ms}
is_zip = src.lower().endswith(".zip")
zf = zipfile.ZipFile(src) if is_zip else None
for nm in names:
    if is_zip:
        arr = np.asarray(Image.open(io.BytesIO(zf.read("output_dir/%s.jpg" % nm))).convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(Image.open(Path(src) / ("%s.jpg" % nm)).convert("RGB"), dtype=np.uint8)
    t = torch.from_numpy(arr.copy()).permute(2,0,1).unsqueeze(0).to(dev).float().div_(255)
    with torch.no_grad():
        for k in ms:
            t0 = time.time(); v = float(ms[k](t).item()); res[k].append(v)
            print("   %-6s %-6s = %.4f (%.0f s)" % (nm, k, v, time.time()-t0), flush=True)
    del t
out = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
out["tag"] = tag; out["dir"] = src
Path(r"D:\models\opt11\e10b_%s.json" % tag).write_text(__import__("json").dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("[E10b] %s -> %s" % (tag, out), flush=True)
print("E10B_DONE", flush=True)
