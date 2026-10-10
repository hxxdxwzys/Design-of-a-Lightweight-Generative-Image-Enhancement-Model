# DOWNLOAD_NEWSRC.md - 新一步扩散复原源下载清单 (2026-08-25)

## 背景
VOSR 家族 (os/ms/1.4B x LoRA) 已全系收束于 M24 (256.3045)。唯一进攻方向 = 新 Diffusion 生成器入池。
门槛: val clip >= 0.90 且 lpips <= 0.376 (新家族转移比未知, 先宽松后严); 架构必须 Diffusion/Flow 系 (竞赛硬性要求, VAR 系有合规风险)。

## 候选排序 (按 EV)

### 1. OFTSR (ICLR 2026) - 首选
- 全称: One-Step Flow for Image Super-Resolution with Tunable Fidelity-Realism Trade-offs
- 为什么首选: **可调保真-真实感旋钮** = 直接对应我们 clipiqa/lpips 前沿问题; rectified-flow 一步 (Diffusion 合规); 小模型流 (4060 可跑)
- 论文: arxiv 2412.09465 | 代码: https://github.com/yuanzhi-zhu/OFTSR
- 需下载: 仓库 (git clone 或 zip) + 权重 (README 里的 Google Drive/HF 链接, 用户浏览器确认具体文件)
- 放置: D:/models/new_src/oftsr/
- 检查项: ① 权重文件名/大小 ② 是否支持 1x (SR 模型常只做 4x) ③ 推理脚本输入规格

### 2. FiDeSR (CVPR 2026) - 高保真设计
- 全称: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution
- 为什么: 论文标题就是'高保真'= 我们的 lpips 主轴; 一步扩散 (合规)
- 论文: arxiv 2603.02692 | 代码: https://github.com/Ar0Kim/FiDeSR
- 需下载: 仓库 + 权重 (README 确认, 用户浏览器)
- 放置: D:/models/new_src/fidesr/

### 3. ODTSR (CVPR 2026) - 权重最确定
- 全称: One-Step Diffusion Transformer for Controllable Real-World Image Super-Resolution (SJTU)
- 为什么: **HF 权重已确认存在** (https://huggingface.co/double8fun/ODTSR); 'controllable' 可能含保真强度控制; 一步扩散 DiT (合规)
- 论文: arxiv 2511.17138 | 代码: https://github.com/RedMediaTech/ODTSR
- 需下载: HF 权重 (double8fun/ODTSR, 浏览器点下载或 huggingface-cli) + 代码仓库
- 放置: D:/models/new_src/odtsr/

### 4. VARestorer (ICLR 2026) - 备选 (合规风险)
- One-Step VAR Distillation for Real-World Image SR; VAR = 自回归非扩散 -> 竞赛'必须 Diffusion'规则存疑
- 代码: https://github.com/EternalEvan/VARestorer | HF: https://huggingface.co/EvanEternal/VARestorer
- 仅当 1-3 全败且用户认可合规解读时评估

## 评估流程 (权重到位后, 我执行)
1. 装依赖/适配 (diffusers 版本坑参考 CODiff 经验)
2. val 3 张 1x 渲染 (align wavelet) -> val_gate
3. 判定: val clip >= 0.90 且 lpips <= 0.376 -> 测试集 100 -> 入池 -> choice -> 打包 -> 呈报
4. 重点检查 1x 模式: SR 模型 4x 输入 1x 图会内容漂移 (OSEDiff 教训); 若只支持 4x: 4x 出图后 2x 下采样 (流形内操作, 参考旧 '逐图 1x/2x' 思路) 或放弃

## 风险提示
- 重度退化训练画像 vs 测试集轻度退化: CODiff 教训 (clip 0.405/lpips 0.582) - 门槛前置防浪费
- 决赛现场复现: 新源出图需同源 model_dir 才能在决赛 20 张复现 (一步模型 512 测速 ~50-150ms, 速度项会降但可接受, 待用户定夺)
