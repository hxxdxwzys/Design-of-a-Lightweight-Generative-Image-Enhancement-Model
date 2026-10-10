import os
d = r"D:\models\待定_解耦探测"
out = r"C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline\_risk_note_copy.txt"
lines = []
for f in sorted(os.listdir(d)):
    lines.append("ENTRY " + ascii(f) + " size=" + str(os.path.getsize(os.path.join(d, f))))
    if f.lower().endswith(".md"):
        raw = open(os.path.join(d, f), "rb").read()
        lines.append("first bytes: " + ascii(raw[:40]))
        for enc in ("utf-8", "gb18030", "utf-16"):
            try:
                txt = raw.decode(enc)
                lines.append("decoded with " + enc + " len=" + str(len(txt)))
                lines.append(txt[:2500])
                break
            except Exception as e:
                lines.append(enc + " failed: " + str(e)[:60])
open(out, "w", encoding="utf-8").write("\n".join(lines))
print("written", os.path.getsize(out))
