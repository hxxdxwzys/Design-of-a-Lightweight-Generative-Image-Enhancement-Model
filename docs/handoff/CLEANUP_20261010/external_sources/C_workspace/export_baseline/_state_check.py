import os
arts = {
 "DELIV02 zip": r"C:\models\exports\DELIV02_20260918_073810",
 "OPT01 zip": r"D:\models\opt01\Track2_OPT01_fp16full_v1.zip",
 "OPT10 zip": r"D:\models\opt10\Track2_OPT10_fp16full_unsharp_v1.zip",
 "OPT10 evidence": r"D:\models\opt10\evidence",
 "iqa_screen": r"D:\models\iqa_screen",
 "base VOSR_0.5B_os": r"D:\models\finals\preset\ckpts\VOSR_0.5B_os",
 "lora v7 s5000": r"D:\models\vosr_lora\v7\lora_s5000.pt",
 "finals dir": r"D:\models\finals",
 "frozen_mode.py": r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline\frozen_mode.py",
 "opt10_gen.py": r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline\opt10_gen.py",
}
for k, p in arts.items():
    if os.path.isdir(p):
        n = sum(len(f) for _, _, f in os.walk(p))
        tot = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs) / 2**30
        print(f"{k:20s} DIR  n={n:5d}  {tot:8.2f} GiB  {p}")
    elif os.path.exists(p):
        print(f"{k:20s} FILE {os.path.getsize(p):>15,} B      {p}")
    else:
        print(f"{k:20s} MISSING                   {p}")
