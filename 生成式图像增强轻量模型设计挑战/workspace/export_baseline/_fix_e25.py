import io, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = r"D:\models\finals_20260927\Track2_light_diffusion_inference\使用说明.md"
s = io.open(D, encoding="utf-8").read()
old = "**验证集 3 对（4K 原生，pyiqa 全套 8 指标）**"
print("锚点存在:", old in s)
if old in s:
    i = s.index(old)
    j = s.index("\n", s.index("。", i)) if "。" in s[i:] else len(s)
    seg = s[i:j]
    new = ("**验证集 3 对（4K 原生）**：我们自建了本地自查口径（有参 PSNR/SSIM/LPIPS/DISTS + 无参 "
           "CLIPIQA/MUSIQ/MANIQA/NIQE），仅用于**内部自查与相对比较**。**官方感知指标的具体构成与权重以官方公布为准**，"
           "本包不对官方口径作任何假设。")
    s = s[:i] + new + s[j:]
    io.open(D, "w", encoding="utf-8").write(s)
    print("已更正材料措辞")
note = io.open(r"D:\models\opt11\_audit_e25.txt", encoding="utf-8").read()
A = r"C:\生成式图像增强轻量模型设计挑战\docs\reference\误判审计台账_20260921.md"
P = r"C:\生成式图像增强轻量模型设计挑战\docs\progress.md"
open(A, "a", encoding="utf-8").write(note)
open(P, "a", encoding="utf-8").write("\n### 41.6 E25：口径更正（用户 2026-09-23 指出）\n\n- 9/1 快照里的「8 指标加权+加速比」是**我们的推测**、非官方机制；官方原文只说「有参考+无参考加权，具体方法决赛入围时公布」。\n- ⇒ 审计结论「官方图像项未知」不变且更强；**禁止**按具体 8 指标向量做定向过拟合；稳健策略 = 不损质量的时延下降 + 多轴同时可见的真实质量提升。\n- 交付材料措辞已同步更正。\n")
print("E25 已写入台账与进度")