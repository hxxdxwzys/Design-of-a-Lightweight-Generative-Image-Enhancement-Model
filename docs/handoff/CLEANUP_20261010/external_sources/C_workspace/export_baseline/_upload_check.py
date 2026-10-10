import os, glob, json
for d in [r"D:\models\upload", r"D:\models\已提交待评测", r"C:\生成式图像增强轻量模型设计挑战\提交历史",
          r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\提交历史"]:
    print("==", d, "exists:", os.path.isdir(d))
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            print(f"   {'DIR ' if os.path.isdir(fp) else 'FILE'} {f}  {os.path.getsize(fp)/2**20:.1f} MiB" if os.path.isfile(fp) else f"   DIR  {f}")
print()
print("== D:\\models\\pkg_* 目录 ==")
for p in sorted(glob.glob(r"D:\models\pkg_*"))[:40]:
    print("   ", p)
