import io, sys, zipfile
from pathlib import Path
from PIL import Image
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ZIP = Path(r"D:\models\upload\Track2_TinyRefV4.zip")
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")
S = Path(r"D:\models\finals_20260927\Track2_light_diffusion_inference\samples")
S.mkdir(parents=True, exist_ok=True)
zf = zipfile.ZipFile(ZIP)
names = [n for n in zf.namelist() if n.lower().endswith(".jpg")]
print("zip 内 jpg:", len(names), "示例:", names[:3])
picks = [n for n in names if n.endswith("case17.jpg")] + [n for n in names if n.endswith("case3.jpg")]
picks = (picks if picks else names[:2])[:2]
for n in picks:
    stem = Path(n).stem
    out_bytes = zf.read(n)
    (S / (stem + "_out.jpg")).write_bytes(out_bytes)
    src = TEST / (stem + ".jpg")
    (S / (stem + "_in.jpg")).write_bytes(src.read_bytes())
    im = Image.open(io.BytesIO(out_bytes)); a = np.asarray(im)
    print("  %-10s 出图 %s  输入存在=%s" % (stem, a.shape, src.exists()))
print("samples ->", S)