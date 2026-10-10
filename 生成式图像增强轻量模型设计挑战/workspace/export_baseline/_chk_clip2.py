import glob, io, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pyiqa
root = os.path.dirname(pyiqa.__file__)
hits = [p for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True) if "clip" in os.path.basename(p).lower()]
print("clip 相关文件:", hits)
for p in hits:
    s = io.open(p, encoding="utf-8").read()
    if "clipiqa" in s.lower() or "CLIPIQA" in s:
        print("====", p, len(s))
        print(s[:1500])
        break