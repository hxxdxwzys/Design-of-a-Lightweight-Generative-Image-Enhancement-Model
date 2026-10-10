import os, glob, json
# 1) pyiqa 指标可用性（本地缓存权重）
cands = ["clipiqa", "niqe", "musiq", "maniqa", "topiq_nr", "clipiqa+", "arniqa", "wadiqam_nr", "brisque", "piqe"]
import pyiqa
ok = {}
for c in cands:
    try:
        m = pyiqa.create_metric(c, device="cpu")
        ok[c] = "OK"
    except Exception as e:
        ok[c] = "FAIL: " + str(e)[:60]
for k, v in ok.items():
    print("%-12s %s" % (k, v))
# 2) 权重缓存位置
for pat in (r"C:\Users\*\ .cache", r"C:\Users\*\AppData\Local\*\hub", r"D:\models\*\hub"):
    pass
for base in [os.path.expanduser("~/.cache"), os.path.expanduser("~/.cache/huggingface"), os.path.expanduser("~/.cache/torch")]:
    if os.path.isdir(base):
        n = sum(len(f) for _, _, f in os.walk(base))
        print("cache:", base, "files:", n)
