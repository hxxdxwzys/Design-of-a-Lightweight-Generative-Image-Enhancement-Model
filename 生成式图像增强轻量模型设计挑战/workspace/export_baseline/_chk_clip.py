import io, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pyiqa, os
p = os.path.join(os.path.dirname(pyiqa.__file__), "models", "clipiqa_model.py")
print("path:", p, os.path.exists(p))
src = io.open(p, encoding="utf-8").read()
print(src[:2600])