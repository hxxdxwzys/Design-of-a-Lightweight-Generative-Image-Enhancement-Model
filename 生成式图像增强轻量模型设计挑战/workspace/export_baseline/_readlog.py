from pathlib import Path
p = Path(r"D:\models\opt11\eval_v18_s25000.log")
print("exists:", p.exists(), "size:", p.stat().st_size if p.exists() else 0)
if p.exists():
    raw = p.read_bytes()
    for enc in ("utf-8", "gb18030"):
        try:
            txt = raw.decode(enc)
            print("--- decoded as", enc, "---")
            print(txt[-1800:])
            break
        except Exception as e:
            print(enc, "fail", e)
