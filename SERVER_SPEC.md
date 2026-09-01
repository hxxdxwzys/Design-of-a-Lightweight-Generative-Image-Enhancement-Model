# 服务器采购/租赁规格单 (2026-09-01)

> 用途: 赛题二终极冲刺 —— VOSR_1.4B LoRA 训练 (patch1024, fp16) + 4K 分块渲染
> 工期: 7-14 天 (2026-09-15 截止)。可直接复制本文件内容发给客服询价。

## 推荐配置 (单卡 A100-PCIE-40GB)

| 部件 | 规格 | 说明 |
|---|---|---|
| **GPU** | **1× NVIDIA A100-PCIE-40GB** | 必选项。要求 40GB 显存、PCIe 4.0 x16；Ampere sm_80 |
| CPU | 16 vCPU (最低 8) | 训练时 PIL 退化/数据增强为 CPU 密集；8 核会拖慢采样 |
| 内存 | **64GB DDR4/DDR5 (最低 32GB)** | 训练进程 ~10-15GB；渲染/评测需同时驻留多张 4K |
| 系统盘 | 80-120GB SSD | 装系统 + torch 环境 (torch cu126 ≈ 3-4GB) |
| **数据盘** | **≥500GB NVMe SSD (最低 300GB)** | 数据 16GB + 权重 9GB + 训练 ckpt 30-80GB + 输出 ~5GB + 余量 |
| 带宽 | 公网 ≥100Mbps 下行 (推荐 1Gbps) | 需下载 ModelScope 权重 ~7GB + Flickr2K 11.6GB |
| 公网 | **公网 IP + SSH (端口 22 或自定义)** | root 访问 |
| 系统 | **Ubuntu 22.04 LTS** (或 TencentOS/其他 Linux) | 兼容性最稳 |
| 驱动/CUDA | NVIDIA 驱动 ≥550 + CUDA 12.4/12.6 | 与 PyTorch 2.7+cu126 匹配；可自装但预装更省时 |

## 加速配置 (可选, 预算允许时)

- **2× A100-PCIE-40GB**（**无需 NVLink**，两张卡独立跑不同训练点/一张训练一张渲染，流水线并行）
  → 13 点网格从 ~2-3 天缩到 ~1.5 天；100 张 4K 测试集渲染双卡并行 ~10h
- 若同机双卡：要求 2× PCIe 4.0 x16 插槽 + 供电 ≥700W（A100-PCIE 单卡 300W）

## 备选 GPU（A100 无货时按优先级）

| 备选 | 可行性 | 代价 |
|---|---|---|
| **NVIDIA A800-40GB/80GB**（中国特供 A100） | ✅ 完全等同 | 价格可能更高 |
| **H100-PCIE-40GB** | ✅ 更快 | 价格高 |
| **RTX 4090-24GB ×2** | ⚠️ 显存不够 patch1024 batch4（需 ~30GB）；只能 patch1024 batch2 或 patch512 batch4 | 训练慢 ~40%，网格 +1-2 天 |
| **RTX 8000-48GB** | ⚠️ 算力仅 A100 1/5 | 网格 8-15 天，赶不上 9/15，不推荐 |

## 网络访问要求（验收时逐项测）

1. `download.pytorch.org`（装 torch cu126 wheel）
2. `www.modelscope.cn`（拉 VOSR_1.4B_os + Qwen-Image-vae-2d，~7GB）
3. `hf-mirror.com`（拉 Flickr2K 11.6GB + pyiqa 权重，国内/海外均可）
4. `pypi.org`（pip 装 pyiqa/peft/safetensors 等）
5. 海外节点另需：`github.com`（torch.hub dinov2，可绕过，已备离线缓存）
6. SSH 入站通畅（22 或指定端口）

## 计费模式建议（2026-09-01 更新）

- ❌ **整机租赁（包月制）不适合**：客服反馈只按包月（整机配置还远超需求）→ 7-14 天的需求包月太贵。
- ✅ **改走按小时计费的 GPU 共享云平台**（AutoDL/恒源云/矩池云等），按 GPU 时长计费，数据盘可关机保留：
  - AutoDL A100-40G ≈ ¥3.5-4.5/h：网格 13 点 2-3 天 + 主训练 1-2 天 + 渲染 ~1 天 → 总 GPU 时长 ~120-160h ≈ **¥500-700**
  - 预算更紧：4090-24G（¥1.5-2.5/h）单卡只能 patch1024 batch2（~2 倍慢），双卡 ≈ A100 单卡吞吐
  - AutoDL 支持"无卡模式"（≈¥0.1/h）先传数据/装环境，再切卡模式训练，省钱
  - 国内平台 modelscope.cn / hf-mirror.com 直连，AutoDL 另有内网 HF/pip 加速
- 按量计费 + 关机保留数据盘；租期预估 5-9 天（9/15 截止）

## 按小时计费平台清单（2026-09-01）

| 平台 | A100-40G 价格 | 说明 |
|---|---|---|
| **AutoDL** (autodl.com) | ≈ ¥3.5-4.5/h | 国内主流，无卡模式省钱，有 A100-40G/80G、4090、V100；推荐 |
| **恒源云** (gpushare.com) | ≈ ¥3-5/h | 同上，有 A100-40G |
| **矩池云 / 智星云 / 优云智算** | ≈ ¥3-6/h | 备选，按量小时计费 |
| **阿里云 ECS gn7e**（A100-40G 按量） | ≈ ¥25-35/h | 贵，除非有券/企业折扣 |
| **腾讯云 CVM GPU 按量** | V100 ≈ ¥8-12/h | 腾讯无 A100-40G 按量（V100 32GB 也能跑，patch1024 batch2） |
| **RunPod / Vast.ai**（海外） | ≈ $0.8-1.2/h | 需外币支付+代理，A100-40G 便宜但国内不稳 |

> 平台差异不大，关键是：**A100-40G 按小时 + 关机保留数据盘**。机器到手验收清单见下节。

## 验收清单（机器交付时检查）

```bash
nvidia-smi              # 应显示 NVIDIA A100-PCIE-40GB, Driver ≥550
df -h /data             # 数据盘 ≥300GB 已挂载可写
free -g                 # 内存 ≥32GB
curl -sI https://www.modelscope.cn | head -1      # 200
curl -sI https://hf-mirror.com | head -1          # 200
pip install torch --index-url https://download.pytorch.org/whl/cu126   # 安装成功
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# 期望: NVIDIA A100-PCIE-40GB, (8, 0)
```
