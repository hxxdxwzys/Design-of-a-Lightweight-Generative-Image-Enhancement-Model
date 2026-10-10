# -*- coding: utf-8 -*-
"""重渲染 finals 包内的 samples/*_out.jpg（使其与新的 thr=1 model_dir 一致）。
来源：赛题二/测试集/case3.jpg 与 case17.jpg（与 samples/*_in.jpg 逐字节相同，已验证）。
"""
import importlib.util, hashlib, io, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
FIN = Path(r"D:\models\finals_20260927\Track2_light_diffusion_inference")
MD = FIN / "model_dir"
TEST = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\测试集")

spec = importlib.util.spec_from_file_location("rr", str(MD / "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(str(MD))
sys.path.insert(0, str(FIN))
import infer as I
dev = str(R.device)
print("[render] graph=%s device=%s" % (R.graph is not None, dev), flush=True)

for nm in ("case3", "case17"):
    src = TEST / (nm + ".jpg")
    lq = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)
    t0 = time.time()
    res = I.render_4k(R, lq, I.make_feather(I.TILE, I.OVER), dev)
    out = FIN / "samples" / (nm + "_out.jpg")
    Image.fromarray(res.astype(np.uint8)).save(out, quality=98, subsampling=0)
    h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print("   %-8s %.1f s -> %s  %d B  sha256:%s" % (nm, time.time() - t0, out.name,
                                                     out.stat().st_size, h), flush=True)

# 立即用与 selfcheck 相同的判据自验
import io as _io
print("\n[verify] 与 selfcheck 相同判据（渲染 -> JPEG(q98,subsampling=0) -> 与样例逐字节比对）", flush=True)
allok = True
for nm in ("case3", "case17"):
    lq = np.asarray(Image.open(TEST / (nm + ".jpg")).convert("RGB"), dtype=np.uint8)
    got = I.render_4k(R, lq, I.make_feather(I.TILE, I.OVER), dev)
    buf = _io.BytesIO()
    Image.fromarray(got.astype(np.uint8)).save(buf, format="JPEG", quality=98, subsampling=0)
    same = (buf.getvalue() == (FIN / "samples" / (nm + "_out.jpg")).read_bytes())
    allok = allok and same
    print("   %-8s %s" % (nm, "PASS(逐字节一致)" if same else "FAIL"), flush=True)
print("SAMPLES_REBUILD_DONE", "ALL_PASS" if allok else "HAS_FAIL", flush=True)
