# 新会话首轮提示词（2026-08-22，M24=256.3045 锁榜后——**本会话目的：突破 lpips 保真瓶颈**）

先读完 CLAUDE.md 和 HANDOFF.md——务必读完 §16.28~16.30（M20~M25 六连发、模型终态、路线收束）+ §16.23~16.27（B 路线），再读 D:/tmp_timing/ 下的 b_* 脚本与 `b_sweep_v7*.json`。

## 🎯 本会话唯一目的：突破瓶颈（lpips 0.378 保真极限）

**当前最优 256.3045（M24：v7_s5000_sc0.8_seed8，test clip 0.9008 / lpips 0.3789 + v7c），暂列第一（217.35 第二，差 ~39 分）。**

**瓶颈定义（三证据）**：
1. **lpips 0.378 是 VOSR-LoRA 保真极限**：LQ 基线 per-case lpips (0.548/0.295/0.436) vs 模型 (0.503/0.253/0.379)——每图 0.04+ 实质改善后触顶；**case1 恒定 0.50 所有 seed/scale 都动不了**。
2. **seed 微调死**：M25（lpips 0.3778 < M24 0.3789）官方分反低 0.33 → **val 3 张的 0.001 级 lpips 差异 = 噪声（±0.5 分）**。
3. **v8 预测两难**：更弱 w_clip（保真向）→ sc0.9 档 lpips 不够低 / sc0.8 档 clip 掉段（v7 的 clip 已悬：test 0.9005-0.9008 在刀刃线上）。

**突破 = 找到 (test clip ≥0.905 安全, lpips ≤0.374) 的点 = +4 分以上**，或结构性新源（质变）。

## 模型终态（八发实证，勿轻易推翻）
- **官方分 ≈ C + clip 项(0.90+ 饱和, 0.87~0.905 陡峭 ~1300-2100×/unit) − lpips 项(~-400~-1150/unit, 低段更陡但噪声大)**
- **铁律：test clip 必须 ≥0.901**（M24 0.9008 / M25 0.9005 的分差 0.32 证实刀刃线）；饱和区内推 clip 无用（M19 0.9263 = 237.97 < M20 0.9053 = 249.70 全因 lpips）
- 每降 0.01 lpips ≈ +4~9 分（低段更陡）；**只有 ≥0.004 的 lpips 改善才值得一发提交**

## 提交历史（官方分）
M17 185.27 → M18 198.07 → M19 237.97 → M20 249.70 → M21 246.24 → M22 250.30 → M23 255.25 → **M24 256.30** → M25 255.98

## 预算
**新号余 6 + 最早号余 1 = 7 次**。（Remay 2 次已尽：M20/M21；新号已用 4 次：M22-M25）

## 突破候选（EV 排序，需用户确认后动手）
1. **v8 保真向训练**：`train_vosr_lora.py --variant v8 --w-clip 1.5~1.8 --steps 8000~10000 --save-every 1000`（~13h）→ val 门控（wavelet 口径）+ scale{0.8,0.85,0.9} 扫 → 目标 (val clip ≥0.925, lpips ≤0.375) → 测试集直选。**赌点：更长训练 + 弱 clip 目标能否把曲线压到 lpips 0.372-0.375 且 sc0.9 档 clip 安全（test ≥0.905）**。M23/M24 已证明 v7 (2.0, 6000) 把曲线下移了 0.004——v8 再移 0.004 即 +3.5 分。
2. **新开源生成器扫描**（质变可能，EV 中）：InstructIR / MambaIRv2 权重需用户协助下载（DOWNLOAD_FINALS.md 模式）；入池后 val 门控（clipiqa≥0.55 且 lpips≤0.372 旧标准已过时——**新标准：test clip ≥0.90 且 val lpips ≤0.375**）。
3. **v7 系 clip 安全边际补强**：找 test clip ≥0.902 且 lpips ≤0.379 的 seed（seed40 val clip 0.9214/lpips 0.3816 差 0.0027 → -2.4 分，不够格）——低 EV，仅当预算充裕。
4. 持分观望（仅当对手无动作且预算紧张）。

## 资产（全在本地）
- 池 **73 源** `D:/models/sweep_test_out/`（天花板 v7_s5000_sc0.8_seed34 clip 0.9005 / M24 的 seed8 0.9008；sweep_results.json 为权威）
- ckpt：`D:/models/vosr_lora/{v5,v6,v7}/`（v7/lora_s5000.pt 为当前王；**v1/v2/v3 已删**）
- 脚本链：`train_vosr_lora.py`（--variant/--w-clip/--steps/--save-every）、`vosr_lora_infer.py`（--lora-scale/--align_method wavelet/--seed）、`pool_add_source.py`、`compute_choice.py`、`package_submit_v2.py --model-dir D:/models/final_graph7/model_dir`（v7c 测速，纯返回静态张量）
- 扫参模板：`b_sweep_v7.py`（ckpt×scale）、`b_sweep_v7seed.py`/`b_sweep_v7deep.py`（seed 扫，带 GPU 等待）、`b_m21_auto.py`（训练完自动门控打分选点链）
- 标定数据：`b_sweep_v7.json`（v7 18 点）、`b_sweep_v7seed.json`（seed 1-7）、`b_sweep_v7deep.json`（seed 8-25）、`b_sweep_v7deep2.json`（seed 25-75，**seed34=0.3778 最低**）、`lq_val_gate.json`（LQ 基线 per-case）

## 纪律与坑（勿重走）
- 像素级后处理全死（M14/M15）；CNN 重训/w48/f2k 死；lpips 精修死（纯 LPIPS 只崩 clip）；**v6 高 clip 路线死（lpips 0.41 拖累，0.9339 clip 白练）**；**seed 微调死（噪声级）**
- **val lpips 0.001 级差异不可信**：只有 ≥0.004 的改善才值得渲染测试集 + 提交
- **clip 刀刃线 0.901**：v7 系 test 转移比 0.9787（wavelet val × 0.9787）；v5 系 0.985；nofix→test 是 0.976
- 渲染输出目录禁与池目录同名（pool_add_source rmtree）；assemble `map:0.010`；`TORCHDYNAMO_SUPPRESS_ERRORS=1`；pyiqa (0,1) fp32；clipiqa 4K；zip 纯 ASCII；杀进程先备份；**连续 subprocess 间等 GPU 显存 <1GB（CUDA 释放延迟）**；PowerShell 输出缓冲错位是假象
- 环境：`D:/Python/python.exe`（3.13 + torch 2.7 cu126）；4060 8GB；D: 51.5GB / C: 29.8GB 空闲（已大清理）
- 提交由用户操作；每次提交前想清"它提供什么标定信息"

## 你的第一步
读完必读文件 → 核对池/ckpt/磁盘 → 向用户要：① 最新榜单（256.30 是否仍第一、217.35 动向）② 预算确认（新号 6 + 最早号 1？）③ 是否启动 v8 训练（13h 长任务）→ 基于标定重新推导方案 + EV 排序，等用户确认"开始"再动手。
