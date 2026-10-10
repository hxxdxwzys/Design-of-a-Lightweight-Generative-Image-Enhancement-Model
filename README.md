# 生成式图像增强轻量模型设计挑战（赛道二）

> **本文件是项目的唯一入口**：项目是什么、赛题硬约束、评分机制、当前成绩、怎么跑、有哪些坑。
> 协作规则（角色 / 权限 / 纪律）⇒ `AGENTS.md`　｜　做到哪里了 ⇒ `docs/progress.md`　｜　现在做什么 ⇒ `docs/current_task.md`
> 最后更新：2026-09-23 上午（**已进入决赛材料阶段**，见 §2.1）。

---

## 1. 项目一句话

天池竞赛**赛道二：生成式图像增强轻量模型设计挑战**——设计**轻量化 Diffusion 模型**对低质量图做清晰度增强（**1x 同分辨率**，非超分），输入输出均为 4K。
评分 = **感知质量**（有参 PSNR/SSIM/LPIPS/DISTS + 无参 CLIPIQA/MUSIQ/MANIQA/NIQE 加权）+ **推理时延加速比**（以 OSEDiff 为基准）。
初赛截止 **2026-09-20**，**前 10 名进决赛**；决赛 = PPT + **现场处理 20 张新 4K 图**当场算分。

## 2. 当前状态（2026-09-23）

| 项 | 值 |
|---|---|
| 主战场 | 账号 C（新世界，2026-09-10 起） |
| **合规最好（本阶段基准）** | **70.1021** —— `Track2_TinyRefV4.zip`（⚠️ 台账推断映射为 V3，待用户核对，见 §2.1） |
| 同族其余合规件 | 69.4540（V3/V4 之一）、68.6681（TinyV21）、68.6300（V4us0）、66.2813（V5） |
| 解耦件（**已放弃，处在复核风险区**） | 105.6401（M24 图 + v4g runner）—— **不作为提交策略**；解耦两件仍在 `提交历史/` |
| 战略主线 | 已从「图像质量硬实力」收束为 **V4 局部最优**：`P10011` 审计结论 = 无任何实验能证明存在严格更优的合规候选 |

完整成绩台账见 `docs/progress.md`；未决风险见 `docs/review.md`。

## 2.1 当前阶段：决赛预选材料（2026-09-23 起，**硬截止**）

初赛已结束，进入**人工检查复核**阶段。**每赛道前 40 名**进决赛预选名单。

| 截止 | 事项 | 谁做 | 状态 |
|---|---|---|---|
| **9/25 12:00** | **全体成员实名认证**（天池个人中心→认证→支付宝）—— 未完成**取消资格** | **仅用户** | ⬜ 待用户确认 |
| **9/27 21:00** | 上传：**推理工程 + 模型权重 + 环境依赖(requirements) + 一页 PPT 方案简介** | 用户上传 / Agent 备料 | 🟡 材料已备好，PPT 缺队名与联系方式 |
| 10/17–10/18 | 决赛（东莞松山湖）：现场推理 + 答辩 | 用户 | ⬜ |

**材料位置**：`D:\models\finals_20260927\`（`Track2_light_diffusion_inference\` 推理工程 + `Track2_方案简介_一页.pptx`）。

**复核红线（通知原文）**：严禁跨团队参赛、调用商用 AI 服务、使用修图软件。

**已知的两个已修缺陷与一项 UNKNOWN**（详见 `docs/progress.md` §42）：包内 `selfcheck.py` 判据曾必然 FAIL（已改为 JPEG 字节级比对，现 PASS）；时延计测缺 GPU 同步（已补，median 6.56 ms）；「70.1021 属 V4 还是 V3」待用户核对。

## 3. 赛题硬性约束（非显然，务必遵守）

- **必须 Diffusion 架构**（一步蒸馏扩散算合格；纯 CNN/Transformer 复原网络不行）。
- **强制 fp16**。
- **同分辨率复原，非超分**：LQ 与 GT 均为 4K（3072×4096 / 4096×3072）。LQ 退化 ≈ 重 JPEG 压缩 + 噪声 + 模糊。
- **测速与出图解耦**：
  - 测速：输入输出**均 512×512**，单卡，**以 OSEDiff 时延为基准**；`model_dir` 只需 512×512 又快又对。
  - 出图：100 张 **4K 离线**生成，**不计入测速**。
- **提交格式**：`zip ≤ 10GB`；`output_dir/case1.jpg…case100.jpg` + `model_dir/{your_model.pt, runner.py}`。
- **`your_model.pt` 需通过 `torch.jit` 导出**（⚠️ 已知形式偏差：M24 那份是权重字典，见 `docs/review.md` §1.1）。
- **runner 可自定义**：官方 `runner.py` 是 OSEDiff 样例（4 个 JIT + config + tokenizer），**不强制**照抄；但必须保留 harness 调用约定
  `Runner(model_dir)` + `infer(image_tensor, prompt="")`，输入输出 `[1,3,H,W] ∈ [-1,1]`。
- **⚠️ 上传 zip 文件名绝对不能含中文**（2026-08-05 用户实测确认，曾因此浪费 3 次提交机会）。用纯 ASCII，如 `Track2_M24_v3.zip`；全角破折号 "——" 经实测可用。
- **提交节奏**：官方原文写"每周一次"，**用户实测确认实际为每账号约 10 次 + 公告再 +5 次、无每周限制**。
- **提交前必须过 9 项门禁**：`preupload_gate.py`（含对抗性 IQA 遮蔽审计）；命名/结构/体积/接口逐条核。

## 4. 评分机制（我们反演出来的模型，非官方公布）

$$
\text{score} = Q(\text{图像}) \times \left(\frac{r_{official}}{0.78}\right)^{0.198},
\qquad r_{official} = \frac{285.6}{t_{local}(ms) + 2.515}
$$

- **【实测】** 由 4 个"图逐字节相同、只改时延"的官方读数**自由拟合**得到，残差 **±0.01 分**。
- ⇒ **官方加速比上限 = 113.6x**（t→0）；M24 那批图在此上限 ≈ **116.4 分** ⇒ 我们 105.6401 已达 **90.7%**。
- 稳健性：C=0（无上限）被慢端读数否决（SSE 27.4）；C=2 → 142.8x；**C=2.515 → 113.6x（SSE 0.0002）**；C=3 → 95.2x。
- **图像项 Q 的已知性质**：跨家族与原生 4K 的 MANIQA 强相关（`Q ≈ 118.3·maniqa − 15.26`，LOOCV 1.78 / 9 点 5 家族），但**家族内失效** ⇒ 它是相关量、不是因果量。
- **有参指标权重 ≈ 0（甚至为负）**：同速同族只加对抗损失，四项有参**全部变差**而官方 **+8.70**。
- ⛔ **旧工具 `score_pred.py`（加法模型）已证伪，禁用**；只用 `D:/tmp_timing/score_pred2.py`。

## 5. 红线（L1–L6，违反即停）

| 编号 | 红线 | 为什么 |
|---|---|---|
| **L1** | 提交的 100 张图必须**真被增强**（test30 lapvar ≥ 60） | 公告"不接受空模型"；空模型实测只有 2.09/255 差异 |
| **L2** | clipiqa ≥ 0.30 | 同上 |
| **L3** | **同主体**：图与 runner 同源 | 公告第三条"须同一模型主体"。⚠️ **我们两发已提交件与此冲突，属已知未决风险（decisions D11）** |
| **L4** | **不碰官方指标**：不做针对 IQA 的优化/对抗/套利 | 4 次 A 类欺诈标签**全部**落在"指标套利"类 |
| **L5** | 时延只在**空闲态**测；**一个 GPU 任务 = 一个进程** | 并发使训练步时 0.34→1.25 s |
| **L6** | zip 名纯 ASCII + 过 9 项门禁 | 历史浪费 3 次机会 |

## 6. 仓库结构

```
C:\生成式图像增强轻量模型设计挑战\        ← git 根 / 工作目录 / 分支 main
├── AGENTS.md                              ← 双 Agent 协作宪法（角色/权限/纪律）
├── README.md                              ← 本文件：项目入口
├── docs\                                  ← 项目内部协作文档（见 §11）
├── 生成式图像增强轻量模型设计挑战\          ← 官方赛题材料 + 自训代码
│   ├── 生成式图像增强轻量模型设计挑战.txt    ← 官方赛题说明（权威原文）
│   ├── runner.py                          ← 官方 OSEDiff 样例 runner（接口参考）
│   ├── 赛题二\ 测试集\（100 张 4K LQ）/ 验证集\（3 对，本地唯一有 GT）
│   └── workspace\                         ← 自训线代码 + eval.py（8 指标闭环）
├── 提交历史\                              ← 已出分的 zip，文件名带 "——分数"
└── DreamLite-main\ neural-enhance-master\ ← 第三方仓库（未改动）
```

**仓库外的关键资产（不在 git 里）**：主力权重 `C:/models/finals/preset/ckpts/VOSR_0.5B_os`；LoRA 家族 `D:/models/vosr_lora/`；合规包与 100 张图 `D:/models/final_compliant/`；工具与台账 `D:/tmp_timing/`；训练数据 `D:/datasets/`。完整索引见 `docs/project_map.md`。

## 7. 快速开始

**解释器：`D:\Python\python.exe`**（torch 2.7.0+cu126 / pyiqa 0.1.16 / CUDA）。⚠️ 系统默认 `python` **没有 numpy**。

本地评测（**必须原生 4K**）：
```bash
cd 生成式图像增强轻量模型设计挑战/workspace
pip install -r requirements.txt        # clipiqa 需 setuptools<81
python eval.py --pred-dir ../赛题二/验证集 --pred-pattern '*_lq.*' \
               --gt-dir   ../赛题二/验证集 --gt-pattern   '*_gt.*' \
               --out results/<name>.json
```

4K 渲染（离线补丁版；**CWD 必须是 `C:\models\finals`**；必须 `TORCHDYNAMO_DISABLE=1`）：
```bash
python D:/tmp_timing/vosr_lora_infer2.py -c C:/models/finals/preset/ckpts/VOSR_0.5B_os \
  --lora D:/models/vosr_lora/v7/lora_s5000.pt --lora-scale 0.8 -i <4K LQ> -o <out> -u 1 \
  --align_method wavelet --seed 8            # 已实测: wavelet = 已提交件的真实配方
```

分数预测：`python D:/tmp_timing/score_pred2.py {table|pred|invert} ...`

## 8. 基线数据（原生 4K，3 对验证集均值）

| 指标 | 方向 | LQ 基线 | GT 天花板 |
|---|---|---|---|
| PSNR | ↑ | 25.79 | — |
| SSIM | ↑ | 0.764 | — |
| LPIPS | ↓ | 0.426 | — |
| DISTS | ↓ | 0.294 | — |
| CLIPIQA | ↑ | 0.251 | 0.462 |
| MUSIQ | ↑ | 24.09 | 40.67 |
| MANIQA | ↑ | 0.302 | 0.388 |
| NIQE | ↓ | 7.67 | 4.79 |

> 注意：**无参指标上我们的生成图已超过 GT**（M24 原生 4K：clipiqa 0.9023 / maniqa 0.5042 vs GT 0.4618 / 0.3883）——这是理解"图像项"的关键事实。

## 9. 易踩坑（历史积累，逐条有效）

- **评测必须原生 4K**，勿加 `--max-side`（缩到 512 会让 PSNR 从 25.79 虚高到 31.13）。
- **clipiqa 依赖 `pkg_resources`**：`setuptools>=81` 已移除 ⇒ 须锁 `setuptools<81`。
- **MUSIQ 在原生 4K 上 51 s/张**（clipiqa 1.3s / maniqa 1.5s / niqe 0.27s）⇒ 面板最多十几张。
- **`vosr_lora_infer.py` 的离线 dinov2 用相对路径** ⇒ 换 CWD 就回落联网并 60s 超时崩；用补丁版 `vosr_lora_infer2.py`。
- **渲染 CLI 的 `ae_path` 也是相对路径** ⇒ CWD 必须是 `C:\models\finals`；`-c` 传的是**目录**（`VOSR_0.5B_os`）。
- **Windows 无 triton** ⇒ 必须 `TORCHDYNAMO_DISABLE=1`（否则 `TritonMissing` 崩）。
- **`model_dir` 桩包会遮蔽官方 pyiqa 依赖** ⇒ 官方报 "Failed to load IQAEvaluator"（runner 内已处理：import 后撤路径 + 清 `sys.modules`）。
- **`measure_dir.py` 的 `--pattern` 必须与渲染后缀一致**（PNG/JPG），否则得到 n=0 的假象。
- **一个 GPU 任务 = 一个进程**；杀进程只按 PID（曾因 PowerShell 条件拼错 `-and` 导致 1.5h 双任务抢卡）。
- **下载类任务自己搞不定要立刻列清单给用户**（文件名/大小/URL/放置路径/安装命令），不要硬下也不要回避。
- **联网依赖是隐性炸弹**：任何 hub/下载调用在断网时会让长链崩在第一步。

## 10. 成绩一览

**新世界（当前，账号 C）**：**合规最好 70.1021**（`Track2_TinyRefV4.zip`，2026-09-20 提交）——**这是本阶段的唯一基准**。
同族合规件：69.4540 / 68.6681 / 68.6300 / 66.2813。⚠️ **解耦件 105.6401 / 94.7249 已被放弃**（公告第三条复核风险，不构成可提交策略）。
完整台账（含 30+ 发逐条）见 `docs/progress.md` §38–§42 与 `docs/reference/新世界模型分数对应表.md`。

**旧世界（历史，部分件已被判欺诈作废）**：

| 版本 | 官方分 | 说明 |
|---|---|---|
| v3 自训 | 20.27 | 56M UNet 一步潜扩散 |
| HYPIR 出图 | 37.52 | 4x-SR 模型 1x 用法 |
| MetricNet1105x | 87.32 | LPIPS-only NAFNet + CUDA Graph ⛔**已被判欺诈检测失败** |
| MetricV7 | 126.63 | s6k 图 + v7c 极限速度 ⛔同批作废 |
| MixS7M18/19 | 198.07 → 237.97 | VOSR-LoRA 感知微调 |
| M24 | 256.3045 | v7_s5000_sc0.8_seed8（旧世界最好） |

## 11. 文档地图

| 去处 | 内容 |
|---|---|
| `AGENTS.md` | **双 Agent 协作宪法**（角色/权限/升级/文档规则/纪律/Git） |
| `docs/project_map.md` | 项目有什么、在哪里（含仓库外资产索引） |
| `docs/architecture.md` | 实际怎么工作（模型/渲染/runner/评分接口/已知架构缺陷 A1–A7） |
| `docs/current_task.md` | 现在做什么 + 验收标准 + 阻塞 |
| `docs/decisions.md` | D1–D16 重要决定与原因（含证据等级） |
| `docs/progress.md` | 成绩台账 / 已完成 / 进行中 / 阻塞 / 已知问题 |
| `docs/review.md` | Architect 审查（当前状态：**尚无正式审查**，有 R1–R5 待裁决） |
| `docs/handoff/` | 历史交接与阶段快照（只增不改） |
| `docs/reference/` | 规则台账 / 分数对应表 / 提交手册 / 公告分析 / 候选调研 |
| `docs/archive/` | 前提已失效的旧文档（引用会误导） |
