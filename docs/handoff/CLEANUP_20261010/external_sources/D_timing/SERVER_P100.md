# SERVER_P100.md — 教学服务器 P100 部署清单 (2026-09-01)

> 沈老师提供的临时账号：2× Tesla P100 16GB（sm_60），相对空闲。
> 定位：**0.5B 家族 patch768/1024 探针**（8GB 卡做不了的"更大 patch"维度），不是 A100 替代。

## 1. ✅ sm_60 兼容性结论（已落实，2026-09-01）
- **torch cu124 线 wheel（2.5/2.6/2.7 +cu124）内置 sm_60 SASS kernel** —— 官方构建矩阵实锤
  （`pytorch-2.5.1-py3.11_cuda12.4_cudnn9.1.0_0 → sm_50, sm_60, sm_61, sm_70, ...`）
- **cu128+/cu129 wheel 已弃 Pascal** → 锁 `+cu124` 后缀，绝不装 cu128
- CUDA 13 / cuDNN 9.11 / TensorRT 10 均弃 Pascal → 全部避开
- 驱动要求：cu124 需驱动 ≥ 550.54（Linux）；**若服务器驱动更老（≥450 即可）→ 退到 cu118 线**
  （torch 2.5.1+cu118 同样含 sm_60 kernel，对老驱动兼容）
- pip 版 torch wheel 自带 cuDNN（≤9.8），无系统 cuDNN 依赖问题

## 2. 拿到账号后 10 分钟验收（先跑这些，再谈安装）
```bash
nvidia-smi                    # 驱动版本 + 2× P100 16GB 确认
python3 --version             # 系统 python（决定装 miniconda 还是用系统）
df -h /                      # 磁盘（需要 ≥50GB）
free -g                      # 内存（≥16GB 即可）
sudo -n true && echo HAS_SUDO || echo NO_SUDO   # 是否有 sudo
```
→ 驱动 ≥550.54：装 torch cu124；驱动 450-549：装 torch cu118；都行则首选 cu124。

## 3. 环境安装（用户级 conda 即可，无需 root）
```bash
# miniconda (用户目录)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p ~/miniconda3 && ~/miniconda3/bin/conda init bash

# torch (cu124 线, 含 sm_60 kernel)
~/miniconda3/bin/pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
# 若驱动老: torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url .../cu118

~/miniconda3/bin/pip install pyiqa peft safetensors pillow numpy requests
~/miniconda3/bin/pip install "setuptools<81"     # clipiqa 依赖 pkg_resources
export TORCHDYNAMO_SUPPRESS_ERRORS=1
```

## 4. 上传清单（本地 → 服务器，~20GB）
| 内容 | 大小 | 说明 |
|---|---|---|
| 代码 `D:/tmp_timing/` 精选 | ~1MB | train_vosr_lora.py（**已打 dinov2 离线补丁**）、train_v2.py、vosr_lora_infer.py、val_gate.py、degradation_analysis.json、jpeg_blockiness.json、b_14b_fp16_align.py |
| VOSR 仓库 `C:/models/finals/VOSR/` | ~50MB | 含 inference_vosr_onestep.py（**已打 dinov2 离线补丁**） |
| VOSR_0.5B_os 权重 | ~2GB | C:/models/finals/preset/ckpts/VOSR_0.5B_os/ |
| SD2.1 VAE + LightDecoder | ~1.5GB | stable-diffusion-2-1-base/vae + sd21_lwdecoder.pth |
| torch_cache（dinov2_vitb14 + pyiqa 权重） | ~1GB | 全离线，服务器不通 GitHub 也能跑 |
| DF2K 数据（div2k 2402 + flickr2k 2650） | ~15GB | 本地上传，或服务器 hf-mirror 拉 |
| 验证集 3 对 4K | ~60MB | 赛题二/验证集 |

## 5. 训练命令（0.5B 双卡并行，两张卡都上 patch1024）
```bash
# 卡0: w_clip 2.0 / patch 1024 / batch 1 / 2500 步
CUDA_VISIBLE_DEVICES=0 nohup python train_vosr_lora.py --variant p1024_w20 \
    --w-clip 2.0 --steps 2500 --save-every 500 --patch 1024 --batch 1 \
    > /data/p1024_w20.log 2>&1 &

# 卡1: w_clip 1.5 / patch 1024 / batch 1 / 2500 步（同 patch 干净对比 w_clip 轴）
CUDA_VISIBLE_DEVICES=1 nohup python train_vosr_lora.py --variant p1024_w15 \
    --w-clip 1.5 --steps 2500 --save-every 500 --patch 1024 --batch 1 \
    > /data/p1024_w15.log 2>&1 &
```
- 速度预估（P100 无 Tensor Core，fp16 ≈ 21 TFLOPS ≈ 4060 的 1/5-1/6）：
  - patch512 ≈ 15-30s/步；patch768 ≈ 40-70s/步；patch1024 ≈ 60-100s/步
  - 2500 步 patch1024 ≈ 40-70h → **双卡并行 2 点 ≈ 2-3 天**（错峰分片跑，脚本支持随时暂停续训）
- 门控：渲染 val 3 张 4K（vosr_lora_infer.py --fp16，P100 上 ~30-60min/张）→ val_gate.py --clip-min 0.919 --lpips-max 0.360
- 决策门同前：过门才渲染测试集；**P100 太慢，不过门绝不全量渲染**

## 6. 纪律
- 训练脚本内置暂停协议（v14_pause.flag 模式，train_vosr_lora.py 已具备）；只用空闲时段
- 杀进程先备份 ckpt json；渲染目录禁与池同名
- 结果 ckpt 回传本地（小文件）→ 本地 4060 渲染/入池（比 P100 快 5 倍）
- 预期管理：这是"降级方案里的最后一个未试维度"，EV 有限；若 patch1024 也没过门 → 0.5B 家族彻底关闭，持分 256.30 等决赛
