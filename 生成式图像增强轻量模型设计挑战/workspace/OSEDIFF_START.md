# OSEDiff 教师复现 — 启动指南 (2026-08-02)

> ⚠️ **结论已定（勿重走）**：OSEDiff **1x 同分辨率评测失败**（clipiqa 0.279/musiq 29.9/psnr 11.1 < v3 0.343/31.5/24.9）。根因：OSEDiff 是 **4x 超分模型**，1x 输入不在训练分布内 → 输出内容漂移（LQ-GT diff 11.3 vs OSEDiff-GT diff 53.9）。直接出图与作教师蒸馏均不可靠。
>
> 本文档保留为"已验证路径记录"（环境搭建、4K 分块推理方法可复用），但**不要再投入开发**。
>
> ⚠️ **下载原则（用户明确要求）**：任何下载需求（依赖装不上、wheel 版本不对、网络不可达），自己下不了**不要硬下也不要回避**——立刻把文件名/大小/URL/放置路径/安装命令**事无巨细**列给用户，用户手动下载。

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
- **torch 必须手动下载 cp311 wheel**（本地 wheel 是 cp313 的; 自动下载会 hash 校验失败）：
  - torch: https://download.pytorch.org/whl/cu126/torch-2.7.0%2Bcu126-cp311-cp311-win_amd64.whl (~2.8GB) → 放 `D:/models/`
  - torchvision: https://download.pytorch.org/whl/cu126/torchvision-0.22.0%2Bcu126-cp311-cp311-win_amd64.whl (~6MB) → 放 `D:/models/`
  - **建议用代理/浏览器下载**（download.pytorch.org 直连不稳）
- **装完后用户告知 Claude → 运行**：
  ```bash
  D:/venv_osediff311/Scripts/pip.exe install D:/models/torch-2.7.0+cu126-cp311-cp311-win_amd64.whl
  D:/venv_osediff311/Scripts/pip.exe install D:/models/torchvision-0.22.0+cu126-cp311-cp311-win_amd64.whl
  # 然后小依赖(清华源):
  D:/venv_osediff311/Scripts/pip.exe install -i https://pypi.tuna.tsinghua.edu.cn/simple diffusers==0.25.0 transformers==4.28.1 peft==0.9.0 loralib fairscale einops==0.7.0 open-clip-torch==2.20.0 huggingface-hub==0.25.0 accelerate==0.20.3
  ```

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
