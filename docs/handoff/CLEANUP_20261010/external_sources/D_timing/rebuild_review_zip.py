# -*- coding: utf-8 -*-
"""按《初赛复核通知》要求重建提交包：
  ① 推理工程 ② 模型权重 ③ 环境依赖(requirements) ④ 一页 PPT 方案简介  -> 【全部打包到一个 .zip】
  不含 output_dir（通知未要求图片）。

关键约束（必须遵守）：
  - 保留 `model_dir/{your_model.pt, runner.py}` —— 这是官方 harness 加载路径，且是历史唯一被平台实测接受的布局。
  - **所有 zip 内部路径纯 ASCII**（历史曾因非 ASCII 内部路径/文件名浪费 3 次提交机会）。
"""
import hashlib, json, shutil, time, zipfile
from pathlib import Path

FIN = Path(r"D:\models\finals_20260927")
SRC = FIN / "Track2_light_diffusion_inference"          # 推理工程（含 model_dir / infer.py / selfcheck.py / 使用说明.md）
OUT = Path(r"D:\models\pkg_review_zip")
ZIP = Path(r"D:\models\upload\Track2_TinyRefV4t1.zip")

if OUT.exists():
    shutil.rmtree(OUT, ignore_errors=True)
(OUT / "model_dir").mkdir(parents=True, exist_ok=True)
(OUT / "samples").mkdir(parents=True, exist_ok=True)

# ① 模型权重（model_dir 内，保持 harness 路径）
shutil.copy2(SRC / "model_dir" / "your_model.pt", OUT / "model_dir" / "your_model.pt")
# ② 推理工程：harness runner 放 model_dir（官方接口）；出图/自检脚本放顶层
shutil.copy2(SRC / "model_dir" / "runner.py", OUT / "model_dir" / "runner.py")
shutil.copy2(SRC / "infer.py", OUT / "infer.py")
shutil.copy2(SRC / "selfcheck.py", OUT / "selfcheck.py")
# ③ 环境依赖
shutil.copy2(SRC / "requirements.txt", OUT / "requirements.txt")
# 说明（改 ASCII 名，避免 zip 内非 ASCII 路径）
shutil.copy2(SRC / "使用说明.md", OUT / "README_CN.md")
# ④ 一页 PPT（改 ASCII 名）
shutil.copy2(FIN / "Track2_方案简介_一页.pptx", OUT / "Proposal_OnePage.pptx")
# 随包样例（支撑"逐字节可复现"的合规声明；体积很小）
for f in (SRC / "samples").glob("*.jpg"):
    shutil.copy2(f, OUT / "samples" / f.name)

# 顶层说明（英文名）
(OUT / "SUBMISSION_README.md").write_text(
    "# Track2 Submission — LiteDiff-Enhance\n\n"
    "Contents (per the review notice):\n"
    "  1) Inference project : `model_dir/`, `infer.py`, `selfcheck.py`\n"
    "  2) Model weights     : `model_dir/your_model.pt` (TorchScript, fp16)\n"
    "  3) Dependencies      : `requirements.txt`\n"
    "  4) One-page proposal : `Proposal_OnePage.pptx`\n"
    "  (extra) `samples/`   : 2 input/output pairs from this exact package, for the byte-exact self-check\n"
    "  (extra) `README_CN.md`: Chinese description (architecture / env / run / compliance)\n\n"
    "Quick check (needs torch + a CUDA GPU):\n"
    "  python selfcheck.py\n"
    "    -> checks the 512x512 harness interface and reproduces the bundled 4K samples\n\n"
    "Harness interface: `Runner(model_dir).infer(image_tensor, prompt=\"\")`, in/out `[1,3,H,W]` in `[-1,1]`.\n"
    "The 4K images are produced by the SAME `model_dir` weights and the SAME code path,\n"
    "with only the patch/overlap configuration differing (explicitly allowed by the rules).\n",
    encoding="utf-8")

# 打包（固定时间戳，纯 ASCII 路径）
FIXED = (2024, 1, 1, 0, 0, 0)
files = sorted([p for p in OUT.rglob("*") if p.is_file()])
assert all(str(p.relative_to(OUT)).isascii() for p in files), "存在非 ASCII 内部路径！"
if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in files:
        rel = str(p.relative_to(OUT)).replace("\\", "/")
        zi = zipfile.ZipInfo(rel, date_time=FIXED)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        z.writestr(zi, p.read_bytes())

print("[zip] %s  %.2f MB" % (ZIP.name, ZIP.stat().st_size / 2**20), flush=True)
with zipfile.ZipFile(ZIP) as z:
    n = z.namelist()
    print("[zip] entries=%d testzip=%s allASCII=%s" % (len(n), z.testzip(), all(x.isascii() for x in n)), flush=True)
    for x in n:
        print("      %-34s %9d B" % (x, z.getinfo(x).file_size), flush=True)

man = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
       "purpose": "初赛复核/决赛预选材料（推理工程+权重+requirements+一页PPT，单一 zip，无图片）",
       "sha256": {str(p.relative_to(OUT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
       "zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
       "zip_bytes": ZIP.stat().st_size}
Path(r"D:\models\opt11\MANIFEST_review_zip.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
print("[zip] zip_sha256 = %s" % man["zip_sha256"], flush=True)
print("REBUILD_REVIEW_ZIP_DONE", flush=True)
