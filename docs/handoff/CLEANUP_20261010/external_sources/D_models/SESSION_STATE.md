# 会话状态与衔接点（SESSION_STATE）

> **用途**：任何时候暂停，下一位 Agent 读这一个文件即可无损衔接。
> **维护者**：DeepSeek Flash。**最后更新**：2026-09-24 18:30。
> 详细读数见 `docs/progress.md` §42.5–§42.19。

---

## 0. 当前在跑 / 未跑（**先读这一节**）

| 项 | 状态 |
|---|---|
| **GPU 占用** | 见下方"活动作业" |
| **活动作业** | **无（全部队列已执行完毕）** |
| **GPU 状态** | 空闲（0 python 进程，~380 MiB，17 W） |
| **全部结论** | 10 条轴全部 CLOSED；唯一正收益 = unsharp thr 3→1。详见 `docs/progress.md` §42.26 |

## 1. 用户已下达的执行顺序（2026-09-24，原话「就按照你列出顺序进行」）

| 序 | 任务 | 类型 | 状态 |
|---|---|---|---|
| 1 | **A1** refiner 损失配方（多尺度判别器） | GPU | ✅ DONE — 负（0.7986） |
| 2 | **B 组** 决赛现场保真度 + 答辩材料 | CPU 并行 | ✅ **DONE**（3 份新文档，见 §7） |
| 3 | **B2** 可学习下采样 | GPU | ✅ DONE — 负（0.5638 < 死亡线 0.63） |
| 4 | **A2** 4K 级跨块一致性 | GPU | ✅ DONE — CLOSED（硬切与羽化 seam_ratio 1.0324 vs 1.0327，指标不可分辨） |
| 5 | **B1** refiner 多步 | GPU | ✅ DONE — CLOSED（迭代发散、残差放大单调递减，单遍最优） |

**分配原则（本会话确立）**：
- **GPU 任务必须串行**（一个 GPU 任务 = 一个进程，并发会使读数失效——见 README §9）。
- **CPU 任务（B 组文档）可与 GPU 任务并行**（实测训练是 GPU-bound）。
- **启动任何 GPU 任务前必须先确认**：`Get-Process python*` = 0 且 `nvidia-smi` 显存接近空载。

## 2. 本会话已确立的硬事实（衔接时必须遵守）

1. **显存墙**：8 GB 卡上，容量超过约 4× v21（3.47M）即触发**换页颠簸**。
   **判据不看利用率、看功耗**：`100% 利用率 + <35 W` = 颠簸；健康值 ≥50 W。
   实测：3.47M/bs8/256域 = 7.1 GiB 健康临界；13.84M/bs8/256域 = 7666 MiB/33 W 颠簸；3.47M/bs4/512域 = 7865 MiB/31 W 颠簸。
2. **时延协议**：必须每次调用前后 `torch.cuda.synchronize()`。不同步会读到 kernel 入队时间（min 0.05 ms，物理不可能）。
   **跨会话的绝对时延不可比**（同一模型不同会话 6.5–14.9 ms）；**只做同会话交替比较**。
3. **口径**：所有质量读数用 `render_subset --down 2 --n 30 --order lex` + `measure_test30`（字典序 30 张、4K 全图 clipiqa/niqe）。**跨口径不可比**。
   B2 用 `b2_eval.py`（同样 30 张、4K 全图、512/32 分块），结果写入 `D:\models\opt11\eval_lrnd\`。
4. **已关闭的轴（9 条，勿重走）**：4K 分块几何、学生容量（单独）、学生容量+refiner（全链）、学生域（512 原生）、refiner 训练步数、refiner 训练域、**refiner 损失配方（多尺度 D）**、所有降质换速、可微 CLIP-IQA 奖励。
5. **唯一严格占优的改进**：内置 unsharp **阈值 3→1**（clipiqa 29/30 胜、niqe 28/30 胜、零时延代价）⇒ `D:\models\archive_candidates\`。
6. **官方图像项 UNKNOWN**，且 **IT-clipiqa 斜率不可辨识**（两对官方读数互相矛盾）⇒ **任何"还有多少空间"的定量结论都必须标明此前提**。
7. **`LearnDown` dtype 陷阱（B2 已踩）**：可学习下采样模块不能用 `.half()`——内部一律 fp32 计算、返回输入 dtype，否则报 `expected scalar type Float but found Half`。
8. **本会话的编辑纪律教训**：用 PowerShell 做代码补丁极易引入**缩进/注释截断/BOM** 三类错误。**一律改用 Python 写补丁器 + `compile()` 校验**，并在长跑前先做 **200 步冒烟测试**（B2 靠冒烟抓到了 dtype 错误）。

## 3. 关键资产位置

| 资产 | 路径 |
|---|---|
| 主力学生 ckpt | `D:\models\tiny_v1\v21_s21000.pt`（0.87M） |
| 已提交 V4 的 refiner | `D:\models\refine_v1\refiner_thin2_s6000.pt`（16ch/2bl） |
| refiner 训练数据 | in=`D:\models\refine_v1\in`（2839，v21 学生输出）；tgt=`D:\datasets\distill256\tgt`（2839，M24 输出） |
| 学生训练数据 | `D:\datasets\distill256\{lq,tgt}`（2839 对 512×512） |
| 决赛交付材料 | `D:\models\finals_20260927\`（**当前为 thr=3 的旧包**，见 §4 待办） |
| 未提交最优候选 | `D:\models\archive_candidates\Track2_TinyRefV4t1_未提交候选.zip`（thr=1） |
| 新候选包（thr=1） | `D:\models\opt11\Track2_TinyRefV4t1.zip`（9 项门禁 PASS） |
| 工具目录 | `D:\tmp_timing\`（见 §5） |

## 4. 待办与阻塞

- **[待用户裁决]** `D:\models\finals_20260927\` 是否换为 thr=1（用户选 B 保持现状；但该包的 `selfcheck.py` 报告的是 thr=3 的 0.8042，**不是我们已知最优**）。
- **[阻塞项]** `Track2_方案简介_一页.pptx` §⑤ 的**队伍名与联系方式**仍是空白下划线 —— **只有用户能提供**。
- **[硬截止]** 9/27 21:00 前上传决赛预选材料；实名认证用户已完成。

## 5. 工具清单（本会话新增，均为通用化副本，**原件未动**）

| 工具 | 用途 |
|---|---|
| `lat_honest.py` / `lat_ckpt.py` | 512 整链时延（前后 GPU 同步）；`lat_ckpt` 可直接吃任意 ckpt |
| `paired.py` | 配对符号检验（4K 全图 clipiqa/niqe，n=30） |
| `tile_sweep.py` / `tile_measure.py` | 4K 分块几何扫描与度量 |
| `us_sweep.py` | unsharp 三参数扫描 |
| `gen_refine_data2.py` | 任意学生 → refiner 训练输入 |
| `train_refiner_gan2.py` / `train_refiner_msd.py` | refiner 训练（后者含 `--msd` 多尺度判别器） |
| `cap2_chain.py` / `c1p_chain.py` / `c2_chain.py` / `a1_chain.py` / `d512_chain.py` | 各实验的"训练→评测"自动链（**可重复运行，产物目录不同名，不会互相覆盖**） |
| `opt11_export_us.py`（在 workspace/export_baseline） | 导出器，新增 `--sigma/--thr` |
| `slope_audit.py` / `c1p_net.py` / `marginal.py` / `space_map.py` | 分数结构与可辨识性分析 |

## 6. 衔接检查清单（恢复时按序执行）

```powershell
# 1) 确认没有僵尸任务
Get-Process python* -ErrorAction SilentlyContinue | Measure-Object
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
# 2) 读活动作业日志
Get-Content D:\models\opt11\a1_chain.log -Encoding UTF8 -Tail 20
# 3) 读本文件 §0，按下达顺序继续
```

## 7. 决赛材料（B 组产出，2026-09-24）

| 文件 | 内容 |
|---|---|
| `D:\models\finals_20260927\INDEX_决赛材料.md` | 材料总索引、模型身份哈希、上传前检查清单、已知取舍 |
| `D:\models\finals_20260927\决赛现场操作手册.md` | 现场 20 张 4K 的完整步骤、时间预算（≈5 min）、6 条风险处置、离线清单 |
| `D:\models\finals_20260927\决赛答辩手册.md` | 评委 12 问与话术、60 秒主线、**主动披露 6 项**、禁忌语 |

**答辩手册的三条硬要求（不可删）**：
1. **主动披露历史解耦件**（105.6401 / 94.7249）已放弃 —— 不主动说而被查出，比主动说严重得多。
2. **主动披露时延测量曾缺 GPU 同步**（修正前 min 0.05 ms 物理不可能，修正后 5.5–7.4 ms），以及笔记本 GPU 持续满载降频（均摊 10–15 ms vs 单次同步 7 ms）。
3. **不知道就说不知道**：官方指标构成、榜首 217 的机制、官方有参/无参权重 —— 全部标 UNKNOWN，**禁止猜测**。

**仍待用户填写**：`Track2_方案简介_一页.pptx` §⑤ 的**队伍名与联系方式**（目前为空白下划线占位）。
