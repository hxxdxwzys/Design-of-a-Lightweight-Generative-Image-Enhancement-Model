import io
A = r"C:\\生成式图像增强轻量模型设计挑战\\docs\\reference\\误判审计台账_20260921.md"
P = r"C:\\生成式图像增强轻量模型设计挑战\\docs\\progress.md"
open(A,"a",encoding="utf-8").write(io.open(r"D:\\models\\opt11\\_audit_e21.txt",encoding="utf-8").read())
open(P,"a",encoding="utf-8").write("\n### 41.5 E21/E24：可微 CLIP-IQA = 奖励攻击实证（2026-09-22 凌晨）\n\n- clip2（w_clip=20）clip 奖励 +42%（0.55→0.783），测试端 clipiqa@4K **0.8593 历史最高**、musiq 63.05、maniqa 0.454、时延不变 ⇒ 纯代理看是最大突破。\n- **肉眼否定**：clip2 严重色偏 + 色调剪贴；clip1 锐化晕环。**E24 val 4K 全参考否定**：lq 25.79/0.9734、**V4 23.15/0.9547/LPIPS 0.4057（最好）**、E8 22.48/0.9481、clip1 18.64/0.8988、clip2 **10.10/0.4513**。\n- 结论：无参照代理可被刷、不可作训练目标；**V4 在两个独立轴上同时居首**；新增判据「代理提升 <5% 必须配全参考 + 肉眼复核」。\n")
print("E21/E24 recorded")