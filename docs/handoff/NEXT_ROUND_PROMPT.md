# 终极冲刺提示词（2026-08-31 交接，用户已拍板服务器路线）

## 你的任务
初赛 9/15 截止。上一轮已把 4060 路线推到物理极限（M24=256.3045，8 轴全封）。**本轮 = 终极冲刺：拿到 GPU 服务器后，用 1.4B 路线冲击 264-276 分**（逼近/追平榜首 275.90）。若服务器到位：按 Phase 0-4 执行；若不到位：0 提交，持分等决赛。

## 现状（必读，勿重走任何已死路线）
- **榜单**：#1 = 275.90（6 发后停滞 = 已到顶；真生成器 lpips≈0.33 或 GT/内部渠道二选一）；**我们 #2 = 256.3045（M24: v7_s5000_sc0.8_seed8）**；老 254.31（8 发，静止）；**新 254.82（5 发，活跃，最大威胁）**。榜单取最优成绩（best-of）。
- **公式终版（13 点 + M26 直接证据）**：Score ≈ C + S(v7c 地板 90-97) + clip(饱和@0.9008) − 400×lpips；其余 6 项权重≈0。**只认 lpips，clip 推升已死**（M26 实测 mean 0.9054 零加成）。
- **已封死（8 轴，全部直接证据，勿重走）**：① 速度=地板 ② clip 饱和@0.9008 ③ lpips 家族地板 0.3789（v5/v6/v7/v8/ms/1.4B-v14/75 种子/尺度/2x/对齐×双骨干/输入预处理全负）④ 其余指标≈0 ⑤ 新源 1x 铁律（5 个 4x-SR 全灭）⑥ GT 私有（EXIF 实锤组织者自摄华为 Pura 80 Ultra）⑦ 像素后处理（M14/M15）⑧ M26 再选择（T 已钉死）。
- **v14 失败点（1.4B 路线必须击败的基准）**：patch512 + w_clip2.0 + 4000 步 → (0.9172, 0.3845)，全部 7 轴输给 M24。失败根因：patch 级 clipiqa 信号不转移 4K、单点赌博、无多阶段。
- **预算**：6 发（新号 5 + 最早号 1；若用户已打 M24 roll ×N 则相应减少）。提交由用户操作。

## 资产
- 权重：VOSR_1.4B_os（C:/models/finals/preset/ckpts/，**qwen VAE 16ch + dinov2l + dim1536/depth36，无 LightDecoder**）；VOSR_0.5B_os；sd21_lwdecoder。
- 代码：`D:/tmp_timing/train_vosr_lora_14b.py`（patch512/8GB 版，**需改 patch1024+batch≥4+fp16 训练**）；`vosr_lora_infer_14b.py`（1.4B 4K 渲染，fp32 22min/图）；`train_vosr_lora.py`（0.5B 版，可作模板）；`val_gate.py`（门控）；`pool_add_source.py`/`compute_choice.py`（池机制）；`package_submit_v2.py`（打包，model_dir=D:/models/final_graph7/model_dir）。
- 数据：DF2K 3450（D:/datasets/div2k/pngtxt_ori 2402 文件 + D:/datasets/flickr2k 2650）；测试退化画像 degradation_analysis.json；val 3 对。
- 1.4B 基座 val：**(0.5403, 0.3322)**，per-case (0.4754/0.1471/0.3741)，dists 0.222，niqe 4.77——全榜唯一 lpips<0.35 的资产。
- 池 73 源（sweep_test_out）；M24/M25 zip 在 提交历史/；v7c 测速件 final_graph7。
- 决赛资产：PPT_OUTLINE.md、FINAL_QUESTIONS.md、20 张演练 23.4min 实测。

## 计划（服务器到位后，按门控推进）

### Phase 0（第 0-1 天）：服务器就绪
- 环境：A100/H100 40GB+（2 卡更佳）、torch cu12x、ModelScope/hf-mirror 拉 VOSR_1.4B_os（或本地上传）；数据上传。
- 改造 train_vosr_lora_14b.py：patch **1024**、batch 4-8、fp16 autocast 训练、存盘每 500 步、暂停续跑协议（v14_pause.flag 模式）。
- **先验证 fp16 渲染与 fp32 对齐**（val 1 张，diff<0.01）→ 1.4B val 门控从 22min→~10min/组合。

### Phase 1（第 1-3 天）⭐ 1.4B 前沿扫描（最高信息量实验，定 265 还是 276）
- 多阶段：**先纯保真阶段**（w_clip=0，LPIPS+L1，把 0.332 再压低）→ 再小 LR 推 clip。
- 网格：w_clip {0.5, 1.0, 1.5, 2.0} × steps {1000, 2000, 4000}（12 点，A100 单卡 ~2-4h/点，2 卡并行 ~1.5 天）。
- 每点 val 门控（fp16 渲染）；**决策门：val (clip ≥0.919, lpips ≤0.360)** → Phase 2；若前沿是陡墙（lpips>0.37@clip0.92）→ 降级：0.5B patch1024 LoRA 试一次（或接受 ~265 天花板，用最优点渲染提交）。
- 对照表：lpips 0.36→263.9 / 0.35→267.9 / 0.34→271.9 / 0.33→275.9（@−400×/unit，均需 val clip≥0.919）。

### Phase 2（第 3-7 天）：主训练 + 种子
- 最优配置全量训练（多阶段，~1-2 天）；种子扫 5-10 个（val 门控，找 seed 运气 ±0.002）；scale {0.8, 0.9} 插值点入池。

### Phase 3（第 7-9 天）：测试集渲染 + 入池
- 1.4B fp16 分块渲染 100 张（~10min/图，2 卡并行 ~8-10h）；逐图 clipiqa → pool_add_source → compute_choice（**注意：clip 饱和 → choice 目标是 lpips 最小，约束 mean clip ≥0.9008；1.4B 若全面支配则无需混合**）。

### Phase 4（第 9-11 天）：提交（1-2 发，呈报用户）
- package_submit_v2（v7c 测速件不变）→ 用户上传。**预算纪律：每发必须过门控（val clip≥0.919 且 lpips 显著 <0.3789）**。

### Phase 5（并行）：决赛准备
- 问主办方决赛规则（FINAL_QUESTIONS.md）；PPT 素材补 1.4B 章节。

## 纪律（照旧）
- 一切训练/渲染带暂停续跑（fidesr_pause.flag 协议）；渲染目录禁与池同名；杀进程先备份 json；zip 纯 ASCII。
- 每发提交前问"门控过没、提供什么标定"；M26 教训：**clip 推升零价值，只打 lpips**。
- 结论写入 HANDOFF §16.5x + STRATEGY §十二。
- 目标诚实：276 = GT 天花板（若 275.90 是 GT 渠道则谁也赢不了，打平即终局）；270-276 = 现实目标；264+ = 保底。
