"""生成 HYPIR 微调的 div2k 文件列表 (只列 png, 空 prompt)。"""
import glob

pngs = sorted(glob.glob(r"D:/datasets/div2k/pngtxt_ori/*.png"))
out = r"D:/datasets/div2k/pngtxt_ori/train_list.txt"
with open(out, "w", encoding="utf-8") as f:
    for p in pngs:
        f.write(p.replace("\\", "/").split("/")[-1] + "\n")
print(f"{len(pngs)} images -> {out}")
