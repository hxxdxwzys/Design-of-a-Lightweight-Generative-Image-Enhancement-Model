led = r"C:\生成式图像增强轻量模型设计挑战\docs\reference\规则细则与判例台账.md"
sc  = r"C:\生成式图像增强轻量模型设计挑战\docs\reference\新世界模型分数对应表.md"
open(led, "a", encoding="utf-8").write(open(r"D:\models\opt11\_led.txt", encoding="utf-8").read())
open(sc,  "a", encoding="utf-8").write(open(r"D:\models\opt11\_sc.txt",  encoding="utf-8").read())
print("appended -> 台账 §8.6 + 分数表 §五")