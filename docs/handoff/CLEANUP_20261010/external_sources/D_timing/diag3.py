# -*- coding: utf-8 -*-
"""验证：新鲜渲染数组 -> JPEG q98/subsampling=0 -> 解回，能否复现已提交出图。"""
import io, os
import numpy as np
from PIL import Image

S = r"D:\models\finals_20260927\Track2_light_diffusion_inference\samples"
exp = np.asarray(Image.open(os.path.join(S, "case3_out.jpg")).convert("RGB"), dtype=np.uint8)
raw = np.load(r"D:\tmp_render_A.npy")
print("raw", raw.shape, raw.dtype, "exp", exp.shape)


def cmp(n1, x, n2, y):
    dd = np.abs(x.astype(np.int16) - y.astype(np.int16))
    print("  %-34s vs %-26s max=%3d mean=%.4f pct!=0=%.2f%%" % (
        n1, n2, dd.max(), dd.mean(), (dd != 0).mean() * 100), flush=True)


u8 = raw.astype(np.uint8)
cmp("fresh render (uint8)", u8, "submitted jpg decoded", exp)

# 1) PIL in-memory round trip (identical encoder args as render_pkg)
buf = io.BytesIO()
Image.fromarray(u8).save(buf, format="JPEG", quality=98, subsampling=0)
rt = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.uint8)
cmp("fresh -> jpg q98 subs0 -> decode", rt, "submitted jpg decoded", exp)
print("     (round-trip vs raw):", end=" ")
cmp("roundtrip", rt, "raw", u8)

# 2) what does the submitted file look like? check its JPEG params
im = Image.open(os.path.join(S, "case3_out.jpg"))
print("  submitted jpg info:", im.info.get("jfif"), "quant tables:", len(getattr(im, "quantization", {}) or {}))
if getattr(im, "quantization", None):
    for k, v in im.quantization.items():
        print("    table %s first8=%s" % (k, list(v)[:8]))

# 3) does encoding raw at other qualities match better?
for q in (95, 96, 97, 98, 99, 100):
    b = io.BytesIO()
    Image.fromarray(u8).save(b, format="JPEG", quality=q, subsampling=0)
    r = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.uint8)
    d = np.abs(r.astype(np.int16) - exp.astype(np.int16))
    print("  q=%-3d size=%9d  vs submitted: max=%3d mean=%.4f pct!=0=%.2f%%" % (
        q, len(b.getvalue()), d.max(), d.mean(), (d != 0).mean() * 100))
