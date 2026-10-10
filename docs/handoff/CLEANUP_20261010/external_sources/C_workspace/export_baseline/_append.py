p = r"D:\models\opt11\prereg_v18_v19.txt"
add = open(r"D:\models\opt11\_outcome_tmp.txt", encoding="utf-8").read()
open(p, "a", encoding="utf-8").write(add)
print("appended", len(add), "chars ->", p)
