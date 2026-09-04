# 新会话首轮提示词 (2026-09-04 交接, 整段可用)

请先完整阅读以下文件再行动：
1. `C:\生成式图像增强轻量模型设计挑战\CLAUDE.md`（项目自述）
2. `C:\生成式图像增强轻量模型设计挑战\HANDOFF.md`（全程记录，重点 §16.53-16.56）
3. `C:\生成式图像增强轻量模型设计挑战\RULE_UPDATE_20260902.md`（规则收紧分析）
4. 本文件（当前任务）

## 项目一句话
天池"生成式图像增强轻量模型设计挑战"（赛道二，初赛 9/15 截止）。**2026-09-02 官方规则收紧：测速必须用与出图同一模型主体，空模型测速（v7c）全部作废，历史成绩面临复核取消。现在是纯质量战场**，速度分 ≈0-6，质量分量 M24≈163 是我们当前最优。预算 11 发提交（全合规）。

## 当前进行中（最高优先级，先检查再行动）
**1.4B VOSR LoRA 双卡续训（2500→6000 步）正在 P100 教学服务器上跑**，~12:30 完成。本地自动验证链（`D:/tmp_timing/p100_watch_local.py` 守护 + `p100_autoval_14b.py`）会自动抓新档（s3000-s6000）渲染门控，结果在 `D:/tmp_timing/p100_val/autoval14_history.txt`。
**你的第一件事：检查续训进度与门控结果**（命令：`D:/Python/python.exe D:/tmp_timing/p100_exec.py "grep -E 'step|save|done' /home/data/tempuser/p100/logs/p512_14b_w20_resume.log | tail -3"`），然后综合判断 1.4B 战略。

## 服务器连接
- 校园网隧道：`10.96.138.171:2222`（tempuser/temp76）—— **需要校园网环境**；域名 bigdata.dynu.net:9077 已被校园网封禁
- 通用执行器：`D:/Python/python.exe D:/tmp_timing/p100_exec.py "命令"`（默认连 10.96.138.171:2222，env P100_HOST/PORT 可覆盖）
- 服务器代码在 `~/p100/`：tools（脚本）/runs（存档）/weights/logs/data
- 守护在岗：服务器 `p100_guard.sh`（训练崩溃自动续训）+ 本地 `p100_watch_local.py`（验证链守护）
- 纪律：只用 GPU0/1（GPU2/3 教学系统勿碰）；PAUSE.flag 秒停协议；教学服务器资源优先

## 1.4B 门控数据基线（首训 2500 步，全 10 档已测）
- w20: s500(0.738,0.351) s1000(0.803,0.364) s1500(0.844,0.384) s2000(0.841,0.380) s2500(0.853,0.388)
- w15: s500(0.736,0.355) s1000(0.786,0.375) s1500(0.819,0.389) s2000(0.837,0.395) s2500(0.845,0.397)
- 门控标准: val clip≥0.919 & lpips≤0.360；决策门过门才渲染测试集
- 关键形态：lpips 起点 0.351（低于 0.5B 家族地板 0.379）但随训练恶化；clip 2500 步仅 0.853

## 战略决策树（续训结果出来后）
1. 若某档 (clip≥0.90, lpips≤0.37)：scale 插值扫（0.7-0.95）→ 找 (clip≥0.919, lpips≤0.360) → 过门 → 100 张测试集渲染 → 合规打包提交
2. 若 clip 平台在 0.85-0.88：试 w_clip 3.0 新训或接受 1.4B 现状 → 与 M24 合规件对比定提交
3. 若续训无改善：1.4B 线收束 → 提交现成合规件 M24（`D:/models/Track2_CompliantM24_Remay_13000000000.zip`，未交）
4. 任何提交前：检查 zip 纯 ASCII、结构（output_dir 100 jpg + model_dir）、Runner 冒烟

## 关键资产位置
- 合规提交件：`D:/models/Track2_CompliantM24_Remay_13000000000.zip`（1.79GB）
- M24 权重：v7 LoRA `D:/models/vosr_lora/v7/lora_s5000.pt`（sc0.8）；合并权重 `D:/models/final_compliant/model_dir/your_model.pt`
- 本地评测：`D:/tmp_timing/val_gate.py`（门控）/ `p100_val/`（渲染输出/历史）
- 服务器工具：train_vosr_lora_14b.py（argparse 参数化，显式传路径）/ p100_resume_14b.sh（续训模板）
- 规则分析：RULE_UPDATE_20260902.md；邮箱草稿在 D:/tmp_timing/email_to_teacher*.md

## 已证伪勿重走（2026-09 追加）
- v7c 空模型测速（违禁）；0.5B 家族全收束（512+768+长训）；patch 增大无收益；像素后处理；闭源 API；speed 剥削

## 需用户提供/确认
- 榜单最新状态（规则复核后是否洗牌）；合规件是否已提交；决赛规则（FINAL_QUESTIONS.md）
