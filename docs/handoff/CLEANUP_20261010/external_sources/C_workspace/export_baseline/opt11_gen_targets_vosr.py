r"""E8 步骤 1：用 OPT-10 打包 runner（VOSR 0.5B + LoRA + unsharp）为 distill256/lq 的 2839 张生成 **VOSR 观感目标**（musiq 65.9 档）。
输出: D:\models\refine_v1\tgt_vosr\*.jpg（q98）
"""
import importlib.util, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
SRC = Path(r"D:\datasets\distill256\lq"); OUT = Path(r"D:\models\refine_v1\tgt_vosr"); OUT.mkdir(parents=True, exist_ok=True)
MD = r"D:\models\opt10\model_dir"
spec = importlib.util.spec_from_file_location("opt10", os.path.join(MD, "runner.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = m.Runner(MD)
fs_all = sorted(SRC.glob("*.jpg"))
fs = [f for f in fs_all if not (OUT / (f.stem + ".jpg")).exists()]
print("N_all = %d, 已有 %d, 本次待生成 %d" % (len(fs_all), len(fs_all)-len(fs), len(fs)), flush=True)
t0 = time.time()
with torch.no_grad():
    for i, f in enumerate(fs, 1):
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        x = torch.from_numpy(a.copy()).permute(2,0,1)[None].float().div_(127.5).sub_(1)
        y = R.infer(x, prompt="")
        o = ((y[0].permute(1,2,0).float().cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
        if o.shape[:2] != a.shape[:2]:
            o = np.asarray(Image.fromarray(o).resize((a.shape[1], a.shape[0]), Image.LANCZOS))
        Image.fromarray(o).save(OUT / (f.stem + ".jpg"), quality=98, subsampling=0)
        if i % 400 == 0: print("  %d/%d %.2f s/张" % (i, len(fs), (time.time()-t0)/i), flush=True)
print("GEN_VOSR_TARGETS_DONE ->", OUT, flush=True)
