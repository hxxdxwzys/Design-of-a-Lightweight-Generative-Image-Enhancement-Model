# 提交通道开放执行手册（SUBMIT RUNBOOK，2026-09-09）

> **本手册不含任何自动提交**：全部脚本只做本地准备与判读（打包/预检/指标/反演），**上传与提交一律由用户手动完成**。
> 工具目录 `D:/tmp_timing/` 里唯一联网的脚本是 SSH 到我们自己的 P100 训练服务器（`p100_*.py`），与赛事网站无关。
> 配套：`NEXT_ROUND_PROMPT_20260909.md`（战略）、`HANDOFF.md §16.68–16.75`、`THEORETICAL_CEILING_FRAMEWORK.md`、`RULE_UPDATE_20260902.md`
> 工具目录：`D:/tmp_timing/`　资产目录：`D:/models/`
> **本手册的唯一目的：通道一开，按判别序列执行时零思考、零返工。**

---

## 0. 触发与前置确认

通道开放后，**先确认三件事再动手**（否则提交可能白费）：

1. **提交通道是否真的开放**（用户页面确认）
2. **剩余提交次数**（欺诈取消是否返还次数；公告 +5 发是否已到账）
3. **欺诈检验名单状态**（哪些历史提交被标、成绩是否保留）

> 历史成绩已按"作废"准备（`HANDOFF §16.68`）。四包资产**从未提交过**，不受复核影响。

---

## 1. 战备状态（2026-09-09 复验，全部就绪）

| # | zip（`D:/models/`） | 大小 | 结构 | 模型 | val (clip, lpips, dists) | 测试集无参 (100 张 clipiqa) | 验证 |
|---|---|---|---|---|---|---|---|
| ① | `Track2_CompliantM24.zip` | 1.67 GB | 100 jpg + 43 文件 | 0.5B v7-sc0.8 | (0.9207, 0.3852, 0.3299) | **0.9008**（std 0.0277） | `HARNESS_SIM_PASS` |
| ② | `Track2_14B_M0S1_SC093.zip` | 3.37 GB | 100 jpg + 44 文件 | 1.4B m0s1-sc0.93 | (0.9261, **0.3571**, 0.3414) | 0.8780（std 0.0373） | `HARNESS_ZIP_SIM_PASS` |
| ③ | `Track2_14B_G3W18S1_SC093.zip` | 3.35 GB | 100 jpg + 44 文件 | 1.4B g3w18s1-sc0.93 | (0.9251, 0.3618, **0.3192**) | 0.8612 | `HARNESS_ZIP_SIM_PASS` |
| ④ | `Track2_14B_M0S7_SC10.zip` | 3.34 GB | 100 jpg + 44 文件 | 1.4B m0s7-sc1.0 | (0.9229, 0.3571, ~0.34) | 0.8790 | `HARNESS_ZIP_SIM_PASS` |

- 本次复验：4 个 zip `testzip()` 全 OK；`output_dir` 各 100 张 jpg；`model_dir` 43/44 文件（自包含 runner，含 `from __future__ import annotations` 低版本兼容修复）。
- 磁盘：C: 44 GB / D: 36 GB 空闲（打包/解压模拟足够）。
- 官方限 10 GB → 最大 3.37 GB，**安全**。

---

## 2. 每发提交 SOP（5 步）

> ⚠️ **解释器坑**：默认 `python`（`C:\Users\...\Python311`）**没有 numpy/torch**，一切脚本必须用
> `D:\Python\python.exe`（3.13 + numpy 2.4.6 + torch 2.7.0+cu126 + CUDA）。

```powershell
# ── 步骤 1｜预检（可选，首次提交或改动后必做）────────────────────────
& D:\Python\python.exe D:/tmp_timing/harness_sim_zip.py D:/models/Track2_CompliantM24.zip  # 期望 HARNESS_ZIP_SIM_PASS
& D:\Python\python.exe D:/tmp_timing/qc_test100.py D:/models/final_compliant/output_dir    # 期望 QC_PASS

# ── 步骤 2｜复制为"纯 ASCII"上传名（严禁中文，全角破折号可用）────────
Copy-Item D:\models\Track2_CompliantM24.zip D:\models\upload\Track2_CompliantM24_Yuki_13000000000.zip
#   命名规范：Track2_<作品名>_<队名>_<电话>.zip（队名/电话按报名信息核对一次）

# ── 步骤 3｜上传（用户在赛事网站操作）────────────────────────────────

# ── 步骤 4｜官方分出现后，回填并自动反演 ────────────────────────────
& D:\Python\python.exe D:/tmp_timing/discriminate.py --record M24=163.0
& D:\Python\python.exe D:/tmp_timing/discriminate.py --report   # Δ 判读 + 下一发建议
& D:\Python\python.exe D:/tmp_timing/f_invert.py               # 最小二乘公式反演（点数≥2）

# ── 步骤 5｜按判读结论决定下一发（见 §3）─────────────────────────────
```

---

## 3. 判别序列（每发必带判别目的）

> 数值来自 `python D:/tmp_timing/discriminate.py`（2026-09-09 实跑）。
> 官方噪声 σ ≈ ±0.4~0.75 → **|Δ| < 6 分视为不可判**。

### 第 ① 发：`Track2_CompliantM24.zip` — 校准弹
- **目的**：新世界第 1 个方程 = 0.5B 质量基准 `C + S(M24)`（旧世界公式认知已作废，必须重建）
- **附带价值**：M24 是测试集无参最强资产（100 张 clipiqa 0.9008，逐图 89:11 碾压 m0s1）→ 若公式含无参权重，它本身就是最优
- **判读**：无（单点只能定标）

### 第 ② 发：`Track2_14B_M0S1_SC093.zip` — lpips 主轴判别（核心）
- **Δ 向量**（m0s1 − M24）：lpips **−0.02812** ｜ test clipiqa **−0.02280** ｜ test musiq −7.50 ｜ test niqe +0.76 ｜ dists +0.0115 ｜ 速度 −1.5
- **情景预测**：
  | clip 饱和线 T | 预测 Δ(m0s1 − M24) | 含义 |
  |---|---|---|
  | T ≥ 0.9008（双方饱和） | **+9.75** | lpips 主轴 |
  | T = 0.90 | +8.69 | lpips 主轴 |
  | T ≤ 0.878（仅 M24 饱和） | **−20.48** | clip 未饱和段主导 |
- **破衡权重**：`w_clipiqa = +427.5/unit`（低于此值 m0s1 赢）；`w_musiq = +1.3`；`w_niqe = −12.8`
- **判读规则**：
  - `Δ ≥ +6` → **公式认 lpips**（clip 对两方饱和）→ **1.4B 主线复活**，下一发走 ③ 或新 1.4B 训练轴
  - `|Δ| < 6` → 不可判/混合权重 → 下一发 ③（dists 三点反演）
  - `Δ ≤ −6` → **clip 未饱和段主导** → **M24 终局**，停止 1.4B 提交，保留预算

### 第 ③ 发：`Track2_14B_G3W18S1_SC093.zip` — dists 判别
- **Δ 向量**（g3w18s1 − m0s1）：dists **−0.02216**（更好）｜ lpips +0.00466（更差）｜ test clipiqa −0.01680
- **破衡权重**：`w_dists = −84.1`（dists 权重更负则 g3 赢）；`w_clipiqa = −111`（正向 clip 权重下 g3 必输）
- **判读**：若观测 Δ ≈ −1.86（lpips 主导预测）→ dists 轴权重≈0；若 Δ 显著为正 → dists 有独立权重（三点反演同时定 lpips/dists/clip 饱和）

### 第 ④ 发：`Track2_14B_M0S7_SC10.zip` — 噪声 σ 测量
- m0s7 与 m0s1 近同位（lpips 同为 0.3571，test clip 差 0.001）
- **预测 Δ ≈ 0.00** → 实测 |Δ| **即官方噪声 σ 的估计**（决定后续"多少分才算显著"）

---

## 4. 失败模式与修复

| 症状 | 原因 | 处置 |
|---|---|---|
| `TypeError: unsupported operand type(s) for \|: 'type' and 'NoneType'` | 官方 Python <3.10，运行时求值 `X \| None` 注解 | 已在 42 个 .py 加 `from __future__ import annotations`（§16.69）。**若再现：把报错原文贴回，我继续修**（下一候选：运行时 `\|` 表达式、typing 运行时依赖） |
| 上传失败 / 文件名报错 | 文件名含中文（曾浪费 3 次机会） | 只用纯 ASCII；全角破折号可用 |
| `Failed to pass the speedup test`（非 TypeError） | 测速环境差异 | 贴回完整报错；本地 harness sim 已 PASS 说明接口正确 |
| 分数迟迟不显示 | 官方队列/复核中 | 记录提交时间，等官方；不重复提交 |
| 显存不足（1.4B 峰值 7.3 GB） | 官方测速卡 <8 GB | 贴回报错；预案 = VAE tiling 优化（`BRAINSTORM §4.1`） |

---

## 5. 每发回传给 AI 的信息清单

1. **提交时间 + 实际文件名**
2. **页面反馈原文**（成功 / 报错全文）
3. **官方分**（出现后）
4. **榜单快照**（我方与前三名分数）
5. **剩余提交次数**

> 我会把 `(版本, 官方分)` 写入 `D:/tmp_timing/asset_profiles.json` → 跑 `discriminate.py --report` 与 `f_invert.py` → 输出下一发建议。

---

## 6. 纪律

- **每发必带判别目的**，没有标定价值的提交不交
- **|Δ| < 2σ 的候选点之间不做重大战略转向**（伪优化）
- 每发结束**立即**把官方分入库（单一数据源：`asset_profiles.json`）
- 历史 v7c 空模型测速路线**永久作废**，任何"提速外挂"不得再进 `model_dir`

---

## 7. 通道未开时的排队任务（按 EV 排序）

| # | 任务 | 成本 | 状态 |
|---|---|---|---|
| 1 | 三个未试训练轴：rank 8/16 收缩 → w_lpips 2–3 → 早切换多阶段 | 各 6.2 h 双卡（P100 空闲） | 🔄 **rank 8/16 已于 2026-09-09 09:56 双卡启动**（`p512_14b_m0s1r8` / `m0s1r16`，ETA ~15:30） |
| 2 | 全 100 张无参升级（4 包 × clipiqa/musiq/niqe） | ~30 min 本地 4060 | 🔄 运行中（`nr_full100_all.py`） |
| 3 | VAE 往返扩测 3 → 9 张（lpips 地板成因确认） | ~1 h | 待排 |
| 4 | 决赛材料去速度神话（`PPT_OUTLINE.md`） | — | 入围后 |

---

*更新纪律：每发提交后更新 §1 表格与 `asset_profiles.json`，并在 `HANDOFF.md` 追加条目。*
