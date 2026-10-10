# 最后一次冲分：完整规划与暂停衔接（FINAL_PUSH）

> 建立 2026-09-24 22:15。**用途**：本文件的 §0 是唯一"当前在哪一步"的权威；任何时刻暂停，下一位读此文件即可无损衔接。
> 配套：`docs/progress.md` §42（十轴读数）、`D:\models\SESSION_STATE.md`（会话级硬约束）。

---

## 0. 当前状态（**先读这一节**）

| 项 | 值 |
|---|---|
| **活动作业** | `pwsh-20` = **S1 两阶段训练**（up1 14k 步 → up1b 6k 步） |
| **日志** | `D:\models\refine_v1\train_up1.log` → 之后 `train_up1b.log` |
| **检查点** | `D:\models\refine_v1\upref_up1_s{N}.pt`（每 2000 步）→ `upref_up1b_s6000.pt` |
| **ETA** | 训练约 75 min（0.24 s/步）⇒ 约 23:30；导出+评测再约 8 min |
| **下一步** | `s1_export.py` → `opt11_render_pkg.py` → `measure_test30.py`（见 §3 命令） |
| **GPU** | 被 S1 占用（98%、51 W 属健康） |

---

## 1. 为什么要"重新审视"（不沿用旧结论）

我在本轮之前关闭了 10 条轴，但**那 10 条都是在同一把滑尺（clipiqa/niqe）下的参数微调**。重新审视发现一个**从未被当作杠杆使用的规律**：

| 底图 clipiqa | refiner 增益（相同 16ch/2b 形状） |
|---|---|
| 0.63（v21 学生） | **+0.174** |
| 0.67（cap2，4× 容量） | +0.130 |
| 0.70（E12） | ~+0.08 |
| **0.90（M24 教师 = refiner 的训练目标）** | — |

**⇒ refiner 的增益随底图变清晰而单调下降。** 而三档容量实验（0.87M / 3.47M / 13.84M）**全部跑在"同等模糊"的底图上**——等于**只扫了容量，从未扫过"底图锐度"这个自变量**。

而底图锐度的**唯一**来源是 `256 →双三次→ 512` 上采样，**它是整条链上唯一无参数、无学习的环节**。
（注：B2 我只换了下采样，没换上采样——两者是不同的环节。）

## 2. S1 实验设计（预登记，已冻结）

| 项 | 值 |
|---|---|
| **假设** | bicubic 上采样造成的模糊是 refiner 的瓶颈；给它**可学习的 2× 上采样器 U**（2188 参数，PixelShuffle，zero-init 残差 ⇒ 起点≈现役），refiner 就能把能力用于**合成**而非**去模糊** |
| **单变量** | 只新增 U；学生冻结；数据/目标/refiner 形状(16ch/2bl)/w_gan/unsharp 与**已提交 V4 的 refiner 配方逐项相同** |
| **训练链** | `512 →(双三次↓256)→ 学生 →(双三次↑512)→ U↑512 → refiner@512`，与打包推理链**逐项一致** |
| **两阶段** | up1：14k 步、w_gan 0.1；up1b：6k 步、w_gan 0.3、lr_d 1e-4（沿用 thin1→thin2 的已验证调度） |
| **判据** | 全链 clipiqa **≥0.83 ⇒ 活**；**>0.8118 ⇒ 净胜当前最优 V4t1**；**≤0.8042 ⇒ CLOSED** |
| **成本** | 训练 75 min + 导出/渲染/评测 8 min |

**为什么这次值得做**：10 条轴里，**没有一条动过"底图锐度"**；而数据显示 refiner 的增益恰恰是底图锐度的函数。这是**唯一一个由数据直接指向、且尚未被触碰的自变量**。

## 3. 执行命令（恢复时照抄）

```powershell
# 0) 确认 GPU 空闲（0 python 进程）
Get-Process python* -ErrorAction SilentlyContinue | Measure-Object
nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader

# 1) 训练（两阶段；若 up1 已完成可只跑第二阶段）
cd D:\tmp_timing
& D:\Python\python.exe -u s1_train.py --tag up1  --steps 14000 --crop 512 --ch 16 --blocks 2 --w_gan 0.1
& D:\Python\python.exe -u s1_train.py --tag up1b --steps 6000 --crop 512 --ch 16 --blocks 2 `
    --w_gan 0.3 --lr_d 1e-4 --init_up D:\models\refine_v1\upref_up1_s14000.pt

# 2) 导出（产出与现役包同接口的 model_dir，可复用已验证的渲染/评测链）
& D:\Python\python.exe -u s1_export.py D:\models\refine_v1\upref_up1b_s6000.pt D:\models\pkg_s1 --thr 1.0

# 3) 渲染 30 张 + 评测（口径与历史 test30 完全一致）
$env:PYTHONIOENCODING="utf-8"
cd C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\export_baseline
& D:\Python\python.exe -u opt11_render_pkg.py --modeldir D:\models\pkg_s1 --out D:\models\opt11\eval_s1 --n 30
& D:\Python\python.exe -u D:\tmp_timing\measure_test30.py --dir D:\models\opt11\eval_s1 --n 30

# 4) 若胜出：配对显著性 + 时延 + 出包 + 门禁
& D:\Python\python.exe -u D:\tmp_timing\paired.py D:\models\opt11\eval_us60_s12_t1 D:\models\opt11\eval_s1
& D:\Python\python.exe -u D:\tmp_timing\lat_honest.py D:\models\finals_20260927\Track2_light_diffusion_inference\model_dir D:\models\pkg_s1
```

## 4. 后续候选（若 S1 为负，按序继续）

| 序 | 实验 | 改动 | 成本 | 为什么 |
|---|---|---|---|---|
| S2 | **U + refiner 同时扩容** | U 16→32ch；refiner 16→24ch | 40 min | S1 若证明"锐度是杠杆"，则应同时扩容以吃满 |
| S3 | **U 在 512 域做多尺度** | U 改为 two-stage（256→384→512） | 40 min | 单次 PixelShuffle 可能不足 |
| S4 | **把 unsharp 换成可学习后处理** | 3 层小 CNN 代替固定核 | 30 min | 固定核已被证明影响 clipiqa 0.006（thr 3→1） |

**已关闭、不要再走**（除非有新证据）：4K 分块几何、学生容量（单独/全链）、学生域（512 原生）、refiner 训练步数/训练域/损失配方、可学习**下**采样、4K 跨块一致性、refiner 多步。

## 5. 纪律（本会话教训，必须遵守）

1. **代码补丁一律用 Python 写补丁器 + `compile()` 校验**（PowerShell 做补丁已三次引入缩进/注释截断/BOM 错误）。
2. **长跑前先做 ≤300 步冒烟**，并同时看 **VRAM 与功耗**（`100% 利用率 + <35 W` = 换页颠簸；健康 ≥50 W）。
3. **一个 GPU 任务 = 一个进程**；启动前确认 `python` 进程数为 0。
4. **时延必须前后 `torch.cuda.synchronize()`**；跨会话绝对时延不可比。
5. **质量口径固定**：字典序 30 张、4K 全图 clipiqa/niqe（`measure_test30`）。**跨口径不可比**。
6. **如实记录负结果与自己的错误**（本轮已三次自我更正：过度描述接缝、误读 D 崩塌、错把"轨迹"当"墙"）。
7. **不得宣称分数上限** —— 官方图像项定义 UNKNOWN，且 IT-clipiqa 斜率不可辨识。
