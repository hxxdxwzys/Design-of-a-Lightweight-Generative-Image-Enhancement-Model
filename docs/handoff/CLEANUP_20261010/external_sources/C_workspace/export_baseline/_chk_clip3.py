import io, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = r"D:\Python\Lib\site-packages\pyiqa\archs\clipiqa_arch.py"
s = io.open(p, encoding="utf-8").read()
i = s.find("class CLIPIQA")
print(s[i:i+3000] if i > 0 else s[2500:5200])