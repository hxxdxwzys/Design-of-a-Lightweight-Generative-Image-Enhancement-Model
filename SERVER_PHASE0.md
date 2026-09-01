# SERVER_PHASE0.md — 终极冲刺服务器就绪清单 (2026-08-31)

> 目标：拿到 GPU 服务器后按本清单在 **0-1 天**内完成 Phase 0，随后启动 Phase 1 前沿扫描。
> 计划文档：NEXT_ROUND_PROMPT.md；决策门：val (clip ≥0.919, lpips ≤0.360) → Phase 2。

## 1. 服务器要求（NEXT_ROUND 已定）
- **A100/H100 40GB+**，2 卡更佳（12 点网格 2 卡并行 ~1.5 天）。
- torch cu12x；Linux（TencentOS/Ubuntu 均可）。
- 香港/海外节点（ModelScope/hf-mirror/GitHub 直连）；若国内节点则全走 hf-mirror.com + modelscope.cn。

## 2. 代码上传清单（本地 → 服务器 /data/proj/tools/）
| 文件 | 用途 | 备注 |
|---|---|---|
| `train_vosr_lora_14b.py` | **1.4B LoRA 训练（Phase0 改造版）** | patch1024/batch4/accum/fp16 autocast/save500/全状态续训/暂停flag |
| `b_14b_fp16_align.py` | fp16 vs fp32 渲染对齐门（diff<0.01） | 先跑这个，过了才能用 fp16 门控 |
| `vosr_lora_infer_14b.py` | 1.4B LoRA 4K 渲染（--fp16） | 门控/测试集渲染 |
| `val_gate.py` | val 3 门控（clipiqa+lpips vs GT） | **已加 --clip-min/--lpips-max/--val-dir 参数**（1.4B 用 --clip-min 0.919 --lpips-max 0.360） |
| `b_14b_frontier.py` | **Phase 1 前沿扫描驱动（12 点网格自动化）** | 每点 train→fp16渲染→gate 全自动 + state 断点续跑 + 孤儿进程防护 + --stop-on-pass |
| `train_v2.py` | EmpProfile 退化采样器 | 训练脚本依赖 |
| `degradation_analysis.json` + `jpeg_blockiness.json` | 测试集退化画像（σ/n/q 三元组） | 训练采样器数据 |
| `dl_vosr_14b.py` / `dl_chunked.py` | ModelScope 权重 / hf-mirror 大文件分块下载 | 服务器自拉权重/数据 |
| VOSR 官方仓库（`D:/models/finals/VOSR/`） | LightningDiT/QwenVAE/dinov2 加载 + inference_vosr_onestep | **⚠️ 已打 dinov2 离线补丁**（见 §5.1），整个目录上传 |

## 3. 权重（服务器优先自拉，ModelScope 直下，带 Range 续传）
```
python dl_vosr_14b.py   # VOSR_1.4B_os/checkpoints/ema_model.safetensors (~6GB) + args.json + Qwen-Image-vae-2d
```
- dinov2_vitl14（torch.hub）：服务器有网时 torch.hub 自动拉（GitHub）；被墙则把本地
  `C:/models/finals/preset/ckpts/torch_cache/` 整个上传。
- pyiqa 权重（LPIPS alex / clipiqa）：pip 装 pyiqa 后首次调用自动下；被墙则上传本地
  `C:\Users\Yuki Nagato\.cache\torch\hub\pyiqa\` 与 `~/.cache/pyiqa/`。
- 权重放置：`/data/weights/VOSR_1.4B_os`、`/data/weights/Qwen-Image-vae-2d`（--ckpt/--qwen-ae 指向这里）。

## 4. 数据（服务器优先自拉，避免上传 15GB）
- Flickr2K 2650 张（11.6GB）：`python dl_chunked.py https://hf-mirror.com/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip /data/datasets/`（已验证可下可续）
- DIV2K 2402 张：hf-mirror 或从本地 `D:/datasets/div2k/pngtxt_ori/` 上传（~3GB）
- 验证集 3 对 4K：上传 `赛题二/验证集/` → `/data/val/`（很小）
- 解压校验：`extract_flickr2k.py` + `validate_imgs.py`（本地已有，一并上传）

## 5. 服务器安装
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # A100/H100 用 cu124/cu126
pip install pyiqa safetensors peft pillow numpy requests
pip install "setuptools<81"          # clipiqa 依赖 pkg_resources
export TORCHDYNAMO_SUPPRESS_ERRORS=1 # 服务器有 triton 则不需要; 无则抑制 dynamo 崩溃
```

### 5.1 ⚠️ dinov2 离线加载补丁（已打，2026-09-01）
- **问题**：`torch.hub.load('facebookresearch/dinov2', ...)` 在检查本地缓存**之前**就会请求
  `api.github.com`（torch/hub.py `_get_cache_or_reload` → `_parse_repo_info` 先行网络校验）。
  GitHub 被墙/抖动时随机超时（实测 17:07 成功 / 17:09 超时）。
- **修复**：`train_vosr_lora_14b.py::load_venc` 与 `VOSR/inference_vosr_onestep.py::load_dinov2`
  均改为**优先本地 hub 缓存仓库**（`torch_cache/facebookresearch_dinov2_main` + `source='local'`，
  权重 `dinov2_vitl14_pretrain.pth` 已缓存 → 完全离线），无缓存再回退远程。
- **服务器注意**：HK/海外节点可直接远程；国内节点把本地 `torch_cache/` 上传即可完全离线。

## 6. Phase 0 验证（半天内）
```bash
# ① fp16 渲染对齐 (本地已验, 服务器上快速重验 1 张):
cd /data/proj/VOSR   # inference_vosr_onestep 相对路径依赖 cwd
python /data/proj/tools/b_14b_fp16_align.py -i /data/val/lq/case1_lq.jpg \
    --lora <任意 v14 存档或训练产物> -o /data/proj/fp16_align_out
# → align_report.txt 必须 PASS (diff<0.01)。通过后 val 门控全程 --fp16 (~10min/组合)。

# ② 训练冒烟 (2 步, 验证环境):
python /data/proj/tools/train_vosr_lora_14b.py --variant smoke --w-clip 0 \
    --steps 2 --patch 256 --batch 2 --save-every 1 --out /data/proj/runs \
    --data-dirs /data/datasets/div2k/pngtxt_ori /data/datasets/flickr2k \
    --ckpt /data/weights/VOSR_1.4B_os --qwen-ae /data/weights/Qwen-Image-vae-2d \
    --deg-json /data/proj/tools/degradation_analysis.json --jpeg-json /data/proj/tools/jpeg_blockiness.json \
    --vosr-dir /data/proj/VOSR --tools-dir /data/proj/tools \
    --pause-flag /data/proj/v14_pause.flag
```

## 7. Phase 1 前沿扫描（服务器到位后主实验，12 点网格）
```
推荐直接跑驱动 (train -> fp16 渲染 -> 门控 全自动, 断点续跑):
  python /data/proj/tools/b_14b_frontier.py \
      --out /data/proj/frontier \
      --data-dirs /data/datasets/div2k/pngtxt_ori /data/datasets/flickr2k \
      --ckpt /data/weights/VOSR_1.4B_os --qwen-ae /data/weights/Qwen-Image-vae-2d \
      --vosr-dir /data/proj/VOSR --tools-dir /data/proj/tools \
      --val-lq /data/val/lq --val-gt /data/val \
      --deg-json /data/proj/tools/degradation_analysis.json --jpeg-json /data/proj/tools/jpeg_blockiness.json \
      --pause-flag /data/proj/v14_pause.flag \
      --parallel 2 --n-gpus 2 --include-prestage --stop-on-pass
手工等价链路:
  阶段 A (纯保真预阶段): w_clip=0, steps 1000, patch1024, batch4, lr 3e-5  # 把 0.3322 再压低
  阶段 B (推 clip):      w_clip {0.5,1.0,1.5,2.0} × steps {1000,2000,4000}  (每点从 A 产物续训或从头)
  每点完成后:
    vosr_lora_infer_14b.py -c /data/weights/VOSR_1.4B_os --lora lora_sN.pt --fp16 \
        -i /data/val/lq -o /data/proj/val_out/<variant> --align_method wavelet
    val_gate.py --src /data/proj/val_out/<variant> --clip-min 0.919 --lpips-max 0.360 --out gate_<variant>.json
决策门: 任一点 (clip≥0.919, lpips≤0.360) → Phase 2 (种子/scale 扫描 + 测试集渲染)
若前沿是陡墙 (lpips>0.37 @ clip0.92): 降级 0.5B patch1024 或接受 ~265 天花板
对照表: lpips 0.36→263.9 / 0.35→267.9 / 0.34→271.9 / 0.33→275.9 (均需 val clip≥0.919)
```
- 每点预估：A100 单卡 ~2-4h（12 点 ≈ 1.5 天双卡并行）。
- 纪律：每点暂停续跑（touch flag → 干净退出）；渲染目录禁与池同名；杀进程先备份 json。

## 8. 提交纪律（预算 6 发，若 roll×3 已打则 3 发）
- 每发必须过门控；提交由用户操作；zip 纯 ASCII。
- 打包链：pool_add_source → compute_choice（目标 = lpips 最小，约束 mean clip ≥0.9008）→ package_submit_v2（model_dir=final_graph7 v7c 测速件不变）。

## 9. 时间线（9/15 初赛截止）
- Day 0：租机 + 本清单 1-5（代码/权重/数据/环境）
- Day 0-1：Phase 0 验证（fp16 align + 训练冒烟）
- Day 1-3：Phase 1 前沿扫描（12 点）→ 决策门
- Day 3-7：Phase 2 主训练 + 种子
- Day 7-9：Phase 3 测试集渲染 + 入池
- Day 9-11：Phase 4 提交 1-2 发
- 服务器不到位 → 0 提交，持分 256.30 等决赛。

## 10. Plan B：预算有限（4090-24GB 单卡，2026-09-01 增补）
> 场景：实验室卡借不到 + 租不起 A100（A100 按量 ~¥3.5-4.5/h 也超出预算）→ AutoDL 4090（~¥2/h）。
> 总成本 ~¥250-300（100-140 GPU 小时）；9/15 前可完成但无缓冲，网格需降级。

- **训练配置**：patch1024 **batch2 + accum2**（显存 ~17-19GB，24GB 装得下；等效 batch4）
- **降级网格（6 点 + 预阶段）**：w_clip {1.0, 2.0} × steps {1000, 2000, 4000} + w0s1000 预阶段
  （A100 的 w_clip{0.5,1.5} 两点砍掉 —— 1.0/2.0 覆盖 v14 邻域即可定前沿形状）
- **节奏预估**（4090 ≈ 12-15s/step）：
  - 网格 7 点 ≈ 60-80h；主训练 4000-6000 步 ≈ 20-35h；100 张 4K fp16 渲染 ≈ 30-40h
  - 合计 ~110-140 GPU 小时 ≈ **¥220-280**；无卡模式（¥0.1/h）做环境/数据准备另算
- **门控不变**：(val clip≥0.919, lpips≤0.360)；渲染照常 --fp16
- **砍项优先级（时间不够时）**：① 种子扫描 5→2；② 主训练 6000→4000 步；③ 测试渲染 q98 直接打包（不做池混合）
