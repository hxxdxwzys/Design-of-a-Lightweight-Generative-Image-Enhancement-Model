import io
A = r"C:\生成式图像增强轻量模型设计挑战\docs\reference\误判审计台账_20260921.md"
P = r"C:\生成式图像增强轻量模型设计挑战\docs\progress.md"
open(A, "a", encoding="utf-8").write(io.open(r"D:\models\opt11\_audit_e25.txt", encoding="utf-8").read())
open(P, "a", encoding="utf-8").write("\n### 41.6 E25：口径更正（用户 2026-09-23 指出）\n\n- 9/1 快照里的「8 指标加权+加速比」是**我们的推测**、非官方机制；官方原文只说「有参考+无参考加权，具体方法决赛入围时公布」。\n- ⇒ 审计结论「官方图像项未知」不变且更强；**禁止**按具体 8 指标向量做定向过拟合；稳健策略 = 不损质量的时延下降 + 多轴同时可见的真实质量提升。\n- 交付材料措辞已同步更正。\n")
print("E25 recorded")