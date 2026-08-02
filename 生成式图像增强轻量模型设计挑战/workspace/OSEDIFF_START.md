# OSEDiff 教师复现 — 新会话启动指南 (2026-08-02)

> 这是**新会话第一条要读的文件**。旧会话已把一切准备妥当，按此执行即可。

## 0. 当前状态快照

- **目标**：跑通官方 OSEDiff 教师（增强专用，非通用 SD2.1），验证 1x 同分辨率 4K 出图是否超 v3（clipiqa 0.343）。超则出 100 张 4K 提交 + v3 测速。
- **v3 保底**：官方 20.27 分，本地复现 0.343，随时可提交（共 10 次机会已用 1 次）。
- **权重全齐**：osediff.pkl + DAPE.pth + RAM(5.6GB) + SD21 base。
- **关键坑**：本地 diffusers 0.39 加载 OSEDiff vae LoRA 输出压平（std 12 vs 官方 50）→ 必须用官方环境（diffusers 0.25）。

## 1. 第一步：搭建官方环境

```bash
cd "C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace"
bash setup_osediff_env.sh
```

- Python 3.11.9 已装好: `C:/Users/Yuki Nagato/AppData/Local/Programs/Python/Python311/python.exe`
- 脚本创建 `D:/venv_osediff311` + 装 diffusers 0.25 官方依赖
- **torch 可能需手动下载 cp311 wheel**（本地 wheel 是 cp313 的）：
  - https://download.pytorch.org/whl/cu126/torch-2.7.0%2Bcu126-cp311-cp311-win_amd64.whl (~2.8GB)
  - https://download.pytorch.org/whl/cu126/torchvision-0.22.0%2Bcu126-cp311-cp311-win_amd64.whl (~6MB)
  - 若脚本报 torch 缺失，让用户手动下这两个文件放 `D:/models/`

## 2. 第二步：跑通官方 test_osediff.py

```bash
VPY="D:/venv_osediff311/Scripts/python.exe"
cd "D:/models/OSEDiff-main/OSEDiff-main"

# 同分辨率 1x (赛题要求): 输入 512 输出 512
"$VPY" test_osediff.py \
  -i preset/datasets/test_dataset/input \
  -o /d/tmp/osediff_test \
  --pretrained_model_name_or_path D:/models/sd21base \
  --osediff_path preset/models/osediff.pkl \
  --ram_path D:/models/ram_swin_large_14m.pth \
  --ram_ft_path preset/models/DAPE.pth \
  --upscale 1 --process_size 512
```

注意：
- `--upscale 1` 是同分辨率（赛题要求 4K 同分辨率增强）；官方默认 4x 超分
- `--ram_path` 指向 RAM 权重, `--ram_ft_path` 指向 DAPE
- 官方环境是 Python 3.11 + diffusers 0.25, 无兼容问题

## 3. 第三步：4K 验证集评测

用官方脚本出 3 张 4K（`--upscale 1` + 输入直接用验证集 LQ），然后：

```bash
cd "C:/生成式图像增强轻量模型设计挑战/生成式图像增强轻量模型设计挑战/workspace"
python eval.py --pred-dir <osediff输出目录> --pred-pattern '*_lq.jpg' \
  --gt-dir "../赛题二/验证集" --gt-pattern '*_gt.*' \
  --out results/osediff_official.json
```

**判定标准**（对比 v3）：
| 指标 | v3 | OSEDiff 需超过 |
|---|---|---|
| clipiqa | 0.343 | >0.343 |
| musiq | 31.5 | >31.5 |

若超 → 出 100 张 4K（`--upscale 1` 分块）+ v3 测速打包提交。
若没超 → 用 OSEDiff 输出做 v5 蒸馏教师（train_v5.py + gen_teacher_targets.py）。

## 4. 备选路径（若官方 OSEDiff 也不理想）

- **DiffBIR**（`task=bid` + `scale=1` 同分辨率增强，权重 hf-mirror 可达）——已验证支持同分辨率。
- **直接提交 v3**（保底 20.27 分，剩 9 次机会）。

## 5. 关键路径速查

| 项 | 路径 |
|---|---|
| OSEDiff 仓库 | `D:/models/OSEDiff-main/OSEDiff-main/` |
| osediff.pkl / DAPE | `.../preset/models/` |
| RAM 权重 | `D:/models/ram_swin_large_14m.pth` |
| SD21 base | `D:/models/sd21base` |
| v3 权重 | `D:/models/unet_v3.pth` |
| v3 提交件 | `C:/生成式图像增强轻量模型设计挑战/submit_v3/` |
| 官方环境 venv | `D:/venv_osediff311` |
| 蒸馏代码 | `workspace/train_v5.py` + `gen_teacher_targets.py` |
| 本地适配版教师 | `workspace/models/osediff_teacher_local.py` (0.39, 仅参考) |
