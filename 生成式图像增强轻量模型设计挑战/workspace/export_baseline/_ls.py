import os
for p in [r"D:\models\opt10", r"D:\models\opt10\model_dir"]:
    print("==", p)
    if os.path.isdir(p):
        for f in sorted(os.listdir(p)):
            fp = os.path.join(p, f)
            print(f"   {'DIR ' if os.path.isdir(fp) else 'FILE'} {f}  {os.path.getsize(fp) if os.path.isfile(fp) else ''}")
