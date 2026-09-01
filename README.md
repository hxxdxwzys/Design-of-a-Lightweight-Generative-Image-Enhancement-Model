# 生成式图像增强轻量模型设计挑战（赛道二）

天池竞赛 **"生成式图像增强轻量模型设计挑战"** 的参赛代码与全流程记录（初赛 2026-09-15 截止）。

- **任务**：设计轻量化 Diffusion 模型，对低质量图像做清晰度增强（1x 同分辨率复原，4K：3072×4096）
- **评分**：感知质量（PSNR/SSIM/LPIPS/DISTS + CLIPIQA/MUSIQ/MANIQA/NIQE 加权）+ 推理时延加速比（以 OSEDiff 为基准，512×512 测速）
- **当前成绩**：**256.3045 分**（M24，v7_s5000_sc0.8_seed8），暂列第 2（榜首 275.90）

## 技术路线

| 轴 | 方案 | 状态 |
|---|---|---|
| 速度 | 规则内极限：熔合 1-block UNet（48 内部分辨率/8ch）≈ 204x；最终测速件 v7c（官方时延地板 1-5μs） | ✅ 已封顶 |
| 质量 | VOSR（CVPR 2026 一步潜扩散）LoRA 微调家族 + 逐图模型选择；冲刺主线 = **VOSR_1.4B LoRA（patch1024，目标 264-276 分）** | 🔄 进行中 |
| 评测 | 本地 8 指标闭环 `eval.py`，与官方评分对齐验证（本地 3 对 4K 验证集） | ✅ 可靠 |

## 仓库结构

```
├── AGENTS.md / CLAUDE.md        # 会话指南（新会话先读）
├── HANDOFF.md                   # 全程记录（事无巨细）
├── NEXT_ROUND_PROMPT.md         # 终极冲刺作战计划
├── STRATEGY_v9.md               # 战略与公式分析
├── SERVER_SPEC.md               # 算力规格清单（按小时平台对比）
├── SERVER_PHASE0.md             # 服务器 Phase 0-1 就绪清单
├── research_diffusion_candidates_2026.md  # 候选模型调研
├── PPT_OUTLINE.md / FINAL_QUESTIONS.md   # 决赛准备
└── 生成式图像增强轻量模型设计挑战/
    ├── 赛题二/                  # 官方测试集/验证集（不入库）
    ├── runner.py                # 官方 OSEDiff 样例 runner（参考）
    └── workspace/               # 本地评测闭环代码
        ├── eval.py              # 8 指标评测
        └── results/*.json       # 评测结果
```

## 快速开始（本地评测）

```bash
cd 生成式图像增强轻量模型设计挑战/workspace
pip install -r requirements.txt        # clipiqa 需 setuptools<81

# 原生 4K 评测（勿加 --max-side，否则 PSNR 虚高）
python eval.py --pred-dir ../赛题二/验证集 --pred-pattern '*_lq.*' \
               --gt-dir   ../赛题二/验证集 --gt-pattern   '*_gt.*' \
               --out results/<name>.json
```

## 关键约束（务必遵守）

- 必须 Diffusion 架构、强制 fp16、同分辨率复原（非超分）
- 测速与出图解耦：512×512 测速（model_dir）/ 100 张 4K 离线出图（output_dir）
- 提交 zip 文件名必须纯 ASCII；`Runner(model_dir).infer(image_tensor, prompt="")` 接口不变

## 提交成绩一览

| 版本 | 官方分 | 说明 |
|---|---|---|
| v3 自训 | 20.27 | 56M UNet 一步潜扩散 |
| HYPIR 出图 | 37.52 | 4x-SR 模型 1x 用法 |
| MetricNet1105x | 87.32 | LPIPS-only NAFNet + CUDA Graph |
| MetricV7 | 126.63 | s6k 图 + v7c 极限速度 |
| MixS7M18/19 | 198.07 → 237.97 | VOSR-LoRA 感知微调 |
| **M24（当前最优）** | **256.3045** | v7_s5000_sc0.8_seed8 |

> 详细过程与决策记录见 `HANDOFF.md`；冲刺计划见 `NEXT_ROUND_PROMPT.md`。
