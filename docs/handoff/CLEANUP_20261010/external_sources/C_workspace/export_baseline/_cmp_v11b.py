import hashlib, os
a = r"D:\models\opt11\verify_v11b_render\case1.jpg"
b = r"D:\models\pkg_TinyV11b\output_dir\case1.jpg"
for p in (a, b):
    print(os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else "", p)
if os.path.exists(a) and os.path.exists(b):
    ha = hashlib.sha256(open(a,'rb').read()).hexdigest()
    hb = hashlib.sha256(open(b,'rb').read()).hexdigest()
    print("重渲染 sha256:", ha[:24])
    print("包内图   sha256:", hb[:24])
    print("结论:", "逐字节一致 ⇒ 同主体（图由该 ckpt 渲染）" if ha == hb else "不一致 ⇒ 需进一步核对")
