r"""E10 代理协议自检：musiq/maniqa 在「4K 全图」vs「512 中心裁切」两种口径下的差异（各 3 张）。
用法: python opt11_e10_protocol.py <zip> [n]
"""
import io, os, sys, time, zipfile
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch, pyiqa
from PIL import Image
TEST = __import__("pathlib").Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
zp = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
dev = "cuda"
ms = {k: pyiqa.create_metric(k, device=dev) for k in ("musiq", "maniqa")}
names = [f.stem for f in sorted(TEST.glob("*.jpg"))[:n]]
res = {"full4k": {k: [] for k in ms}, "crop512": {k: [] for k in ms}}
with zipfile.ZipFile(zp) as zf:
    for nm in names:
        arr = np.asarray(Image.open(io.BytesIO(zf.read("output_dir/%s.jpg" % nm))).convert("RGB"), dtype=np.uint8)
        H, W = arr.shape[:2]
        crop = arr[(H-512)//2:(H-512)//2+512, (W-512)//2:(W-512)//2+512]
        for key, a in (("full4k", arr), ("crop512", crop)):
            t = torch.from_numpy(a.copy()).permute(2,0,1).unsqueeze(0).to(dev).float().div_(255)
            with torch.no_grad():
                for k in ms:
                    t0 = time.time(); v = float(ms[k](t).item()); dt = time.time()-t0
                    res[key][k].append(v)
                    print("   %-8s %-7s %s = %.4f (%.1f s)" % (nm, key, k, v, dt), flush=True)
            del t
for k in ms:
    f = float(np.mean(res["full4k"][k])); c = float(np.mean(res["crop512"][k]))
    print("[E10] %-7s 4K=%.4f  512crop=%.4f  差=%+.4f (%+.1f%%)" % (k, f, c, f-c, 100*(f-c)/c), flush=True)
print("E10_DONE", flush=True)
