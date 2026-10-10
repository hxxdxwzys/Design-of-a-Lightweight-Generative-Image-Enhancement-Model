"""E4 门槛评测: 用指定微调权重在 3 张验证集 4K 上出图 + 评测, 对比 HYPIR 基线。

基线 (HYPIR_sd2.pth): lpips 0.3364 / clipiqa 0.4400 / musiq 36.81 / psnr 22.13
门槛: lpips < 0.3364 且 clipiqa >= 0.44 且 musiq >= 36.8

用法: python tmp_e4_gate.py <weight_path> <out_tag>
"""
import subprocess
import sys
import json
from pathlib import Path

BASE = r"D:/models/hypir_code/HYPIR-main/HYPIR-main"
VAL_LQ = r"D:/models/hypir/lq_val3"
VAL_GT = r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\验证集"
WS = Path(r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace")

wpath = sys.argv[1]
tag = sys.argv[2]
# 优先用 EMA 权重 (更稳定)
wpath_dir = Path(wpath).parent
ema = wpath_dir / "ema_state_dict.pth"
if ema.exists():
    wpath = str(ema)
    print(f"using EMA weights: {wpath}", flush=True)
out_dir = f"D:/models/hypir/gate_{tag}"
res_json = WS / "results" / f"gate_{tag}.json"

cmd = [r"D:\Python\python.exe", "test.py",
       "--base_model_type", "sd2", "--base_model_path", r"D:/models/sd21base",
       "--model_t", "200", "--coeff_t", "200",
       "--lora_rank", "256",
       "--lora_modules", "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
       "--weight_path", wpath,
       "--patch_size", "512", "--stride", "256",
       "--lq_dir", VAL_LQ, "--upscale", "1", "--captioner", "empty",
       "--output_dir", out_dir, "--device", "cuda"]
print(">>", " ".join(cmd), flush=True)
r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr[-2000:])
    sys.exit(1)

cmd2 = [r"D:\Python\python.exe", "eval.py",
        "--pred-dir", f"{out_dir}/result", "--pred-pattern", "*.png",
        "--gt-dir", VAL_GT, "--gt-pattern", "*_gt.*",
        "--metrics", "psnr,ssim,lpips,clipiqa,musiq,maniqa,niqe",
        "--out", str(res_json)]
print(">>", " ".join(cmd2), flush=True)
r2 = subprocess.run(cmd2, cwd=WS, capture_output=True, text=True)
if r2.returncode != 0:
    print(r2.stderr[-2000:])
    sys.exit(1)

d = json.loads(res_json.read_text(encoding="utf-8"))["mean"]
m = {k: round(v, 4) for k, v in d.items()}
print(f"\n=== gate {tag}: {m}")
# 主判据 lpips (官方公式权重最大); 次要指标放宽防误杀 (musiq 地板 35 / clipiqa 地板 0.42)
ok = (m["lpips"] < 0.3364 and m["musiq"] > 35.0 and m["clipiqa"] > 0.42)
print(f"=== PASS(过门槛)={ok}  [主判据 lpips<0.3364; 次判据 musiq>35 clipiqa>0.42]")
sys.exit(0 if ok else 2)
