# 生成式图像增强轻量模型设计挑战（赛道二）— 工作区

## 任务速览
- **同分辨率真实图像复原**（非超分）：LQ 与 GT 均为 4K（3072×4096 / 4096×3072），LQ 为重 JPEG 压缩+噪声+模糊退化。
- **必须 Diffusion 架构**；**强制 fp16**；评分 = 感知指标(有参+无参加权) + 推理时延加速比。
- **测速在 512×512**（以 OSEDiff 时延为基准）；**出图为 4K**（100 张，离线分块生成）。
- 提交：`output_dir/case*.jpg` + `model_dir/{your_model.pt, runner.py}`，zip≤10GB。

## 评测脚本 `eval.py`
有参(PSNR/SSIM/LPIPS/DISTS) + 无参(CLIPIQA/MUSIQ/MANIQA/NIQE)，按 case 号匹配。
```bash
# LQ 基线（pred=lq, gt=gt）
python eval.py --pred-dir ../赛题二/验证集 --pred-pattern '*_lq.*' \
               --gt-dir   ../赛题二/验证集 --gt-pattern   '*_gt.*' \
               --out results/lq_baseline_4k.json
# 模型输出评测（pred=模型输出目录）
python eval.py --pred-dir <输出目录> --gt-dir ../赛题二/验证集 --gt-pattern '*_gt.*' \
               --out results/<name>.json
# 快速冒烟（长边缩到512，仅用于跑通；会掩盖4K退化，不可作正式基线）
python eval.py ... --max-side 512
```
> 注意：**正式评测必须用原生 4K**（`--max-side` 留空）。下采样会掩盖 JPEG 退化，512 下数值严重偏乐观。

## 基线（原生 4K，3 对验证集均值）
| 指标 | 方向 | LQ基线 | GT天花板 | 说明 |
|---|---|---|---|---|
| PSNR | ↑ | 25.79 | — | 有参 |
| SSIM | ↑ | 0.764 | — | 有参 |
| LPIPS | ↓ | 0.426 | — | 有参 |
| DISTS | ↓ | 0.294 | — | 有参 |
| CLIPIQA | ↑ | 0.251 | 0.462 | 无参，提升空间大 |
| MUSIQ | ↑ | 24.09 | 40.67 | 无参，提升空间大 |
| MANIQA | ↑ | 0.302 | 0.388 | 无参 |
| NIQE | ↓ | 7.67 | 4.79 | 无参 |

结论：LQ 在 4K 下退化明显；无参指标离 GT 天花板有较大间距 → 模型主要发力于无参感知质量，同时尽量保住有参保真度。

## 技术路线
图像条件**一步潜扩散**（去 RAM/CLIP）：小 VAE(8×下采样) + 小 UNet，LQ 为条件，1 步去噪。
- 加速比突破口：砍掉 OSEDiff 的 RAM+CLIP 文本路径 + 缩小 VAE/UNet。
- 4K 出图：512 分块带重叠拼接（3072×4096 = 6×8 块）。
- 训练：云 GPU，合成真实退化(JPEG+噪声+模糊+下采样)于 HQ 数据集，或从 OSEDiff 蒸馏。

## 依赖
`pip install -r requirements.txt`（clipiqa 需 `setuptools<81` 以保留 `pkg_resources`）

## 目录
```
workspace/
├── eval.py            # 评测闭环
├── requirements.txt
├── README.md
└── results/           # 评测 JSON
    ├── lq_baseline_4k.json
    ├── lq_smoke.json
    └── gt_ceiling_4k.json
```
