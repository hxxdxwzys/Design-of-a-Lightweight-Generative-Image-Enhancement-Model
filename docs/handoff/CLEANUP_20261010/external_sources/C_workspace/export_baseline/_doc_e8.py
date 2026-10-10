import io
A = r"C:\生成式图像增强轻量模型设计挑战\docs\reference\误判审计台账_20260921.md"
P = r"C:\生成式图像增强轻量模型设计挑战\docs\progress.md"
open(A,"a",encoding="utf-8").write(io.open(r"D:\models\opt11\_audit10.txt",encoding="utf-8").read())
open(P,"a",encoding="utf-8").write(io.open(r"D:\models\opt11\_prog41.txt",encoding="utf-8").read())
print("docs updated: E8 + progress 41")