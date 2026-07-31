# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 本文件是项目的"自我提醒"——任务定义、约束、资源、服务器访问、基线、技术路线与进度。新会话先读这里。

## 项目是什么
**赛道二：生成式图像增强轻量模型设计挑战**。设计**轻量化 Diffusion 模型**对低质量图做清晰度增强，评分 = 感知质量 + 推理时延加速比。

## 关键约束（非显然，务必遵守）
- **必须 Diffusion 架构**（一步蒸馏扩散算合格；纯 CNN/Transformer 复原网络不行）。
- **强制 fp16**。
- **同分辨率复原，非超分**：LQ 与 GT 均为 4K（3072×4096 / 4096×3072），LQ 退化=重JPEG压缩+噪声+模糊。
- **测速与出图解耦**：
  - 测速：输入输出**均 512×512**，单卡测时延，**以 OSEDiff 时延为基准算加速比**。提交的 `model_dir/runner.py + your_model.pt` 只需 512×512 又快又对。
  - 出图：100 张 **4K**，离线分块生成提交 jpg，不计入测速。
- **评分**：有参(对GT: PSNR/SSIM/LPIPS/DISTS)+无参(CLIPIQA/MUSIQ/MANIQA/NIQE)加权 + 加速比。具体权重决赛入围时公布。**每周仅 1 次上传**。
- **提交格式**：`zip≤10GB`，结构 `output_dir/case1.jpg...case100.jpg` + `model_dir/{your_model.pt, runner.py}`。命名 `赛题二_xxx作品_xx队_13000000000.zip`，名中无中文符号。
- **runner 可自定义**：官方 `runner.py` 是 OSEDiff 样例（4 个 JIT：osediff_core/clip_text_encoder/ram_dape_encoder/ram_tag_decoder + config.yaml + ram_tag_params.pth + tokenizer）。**不强制用 4 文件接口**，可写自己的 runner，但保留评测 harness 统一调用的 `Runner(model_dir)` + `infer(image_tensor, prompt="")` 接口（输入输出 [1,3,H,W]∈[-1,1]）。

## 仓库结构
```
C:\生成式图像增强轻量模型设计挑战\           ← cwd / 项目根 (CLAUDE.md 在此)
└── 生成式图像增强轻量模型设计挑战\
    ├── 生成式图像增强轻量模型设计挑战.txt   ← 官方赛题说明（权威）
    ├── runner.py                            ← 官方 OSEDiff 样例 runner（参考）
    ├── 赛题二\
    │   ├── 测试集\  (100 张 LQ jpg, 4K)      ← 出图对象
    │   └── 验证集\  (case1~3 各 _gt+_lq, 4K) ← 本地评测
    └── workspace\                           ← 我们的代码
        ├── eval.py          ← 本地评测闭环（8 指标）
        ├── requirements.txt
        ├── README.md        ← 含基线表与技术路线
        └── results\*.json   ← 评测结果
```

## GitHub 仓库
- 远程: https://github.com/hxxdxwzys/Design-of-a-Lightweight-Generative-Image-Enhancement-Model （建议 Private）
- 本地 git 根 = 项目根 (C:\生成式图像增强轻量模型设计挑战)，分支 main。
- PAT 在 `~/.claude/settings.json` 的 `GITHUB_PERSONAL_ACCESS_TOKEN`，已写入 `.git/config` 的 remote URL（push 免密）。
- `.gitignore` 排除：`赛题二/` 数据、`workspace/submit_v3.zip`(526MB)、`preview/`、`.secrets`(服务器密码/HF token)。
- 提交工作流：改代码 → `git add -A && git commit -m "..."` → `git push`。
- 服务器密码/IP 已从 CLAUDE.md 移至本地 `.secrets`（不入库）。

## 常用命令
```bash
# 本地评测（必须原生 4K，勿加 --max-side）
cd 生成式图像增强轻量模型设计挑战/workspace
python eval.py --pred-dir ../赛题二/验证集 --pred-pattern '*_lq.*' \
               --gt-dir   ../赛题二/验证集 --gt-pattern   '*_gt.*' \
               --out results/<name>.json
# 评测模型输出: --pred-dir <输出目录>  其余同上
# 装依赖: pip install -r requirements.txt   (clipiqa 需 setuptools<81)
```

## 易踩坑
- **评测必须原生 4K**：长边缩到 512 会让 PSNR 从 25.79 虚高到 31.13（下采样掩盖 JPEG 退化）。`--max-side 512` 仅用于跑通冒烟，不可作基线。
- **clipiqa 依赖 `pkg_resources`**：`setuptools>=81` 移除了它 → `requirements.txt` 锁 `setuptools<81`，否则 clipiqa 创建报 `ModuleNotFoundError`。
- 本地 torch 是 **CPU-only**，4K 上跑 lpips/dists/maniqa 较慢（3 对约 5–8 分钟），正常。

## 基线（原生 4K，3 对验证集均值）
| 指标 | 方向 | LQ基线 | GT天花板 |
|---|---|---|---|
| PSNR | ↑ | 25.79 | — |
| SSIM | ↑ | 0.764 | — |
| LPIPS | ↓ | 0.426 | — |
| DISTS | ↓ | 0.294 | — |
| CLIPIQA | ↑ | 0.251 | 0.462 |
| MUSIQ | ↑ | 24.09 | 40.67 |
| MANIQA | ↑ | 0.302 | 0.388 |
| NIQE | ↓ | 7.67 | 4.79 |
→ 无参指标离 GT 天花板间距大，主攻感知质量、保住有参保真度即可显著得分。

## 技术路线（已定）
**蒸馏/裁剪 OSEDiff** → 图像条件**一步潜扩散**（去 RAM/CLIP）：
- 小 VAE（8× 下采样，4~8 通道潜码）+ 小 UNet，LQ 为条件，1 步去噪。
- 加速比突破口：砍掉 OSEDiff 的 RAM+CLIP 文本路径 + 缩小 VAE/UNet。
- 4K 出图：512 分块带重叠拼接（3072×4096 = 6×8 块）。
- 训练在云 GPU；合成真实退化(JPEG+噪声+模糊+下采样)于 HQ 数据集，或从 OSEDiff 蒸馏。

## 云 GPU 服务器（v4 用，上海）
- **腾讯云 CVM**，实例 `GN10X.2XLARGE40`（Tesla **V100 32GB**，8核40G），上海五区，按量计费。
- **公网 IP / 密码 / SSH 端口：见本地 `.secrets`**（端口 **2222**，非 22——22 入站被拦）。
- 镜像 Ubuntu 22.04 + 驱动 580 + CUDA 13 + cuDNN 预装。Python 3.10.12。
- 数据盘 500G 挂载 `/data`。上海区用 `hf-mirror.com` 镜像下 HF 资源（设 `HF_ENDPOINT=https://hf-mirror.com`）。
- ⚠️ 密码在 `.secrets`（不入库）；用完关机或销毁。

## 本地环境
Windows 11，Python 3.13，torch 2.13.0+cpu（CPU-only）。本地只跑评测(3对)与写代码；训练/出图上云。
workspace 路径：`生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\`

## 进度（任务清单）
- ✅ 搭建本地评测闭环（eval.py，8 指标）
- ✅ 建立 LQ 基线 + GT 天花板（4K）
- ✅ 配置并初始化云 GPU 服务器（V100 32GB, 驱动570, torch2.13+cu126, /data 500G）
- ✅ 获取 OSEDiff 仓库(含osediff.pkl/DAPE.pth) + SD2.1 VAE + DIV2K(548张已提取)
- ✅ 设计轻量一步潜扩散架构（SD2.1 VAE冻结 + 56M UNet + 显式x0 t=500，无RAM/CLIP）
- ✅ 训练/蒸馏/出图/打包（**v3=DF2K+PatchGAN 最优**: clipiqa0.344/musiq31.5/lpips0.387/niqe6.56, psnr24.88; v2(clipiqa0.291)/v1(0.296)保底; submit_v3.zip 就绪 526MB, 已验JIT+runner端到端OK）
- ⬜ 复现 OSEDiff 基线 + 测 512 时延（加速比分母，延后；教师权重SD21+RAM已下好备用）
- ⬜ 复现 OSEDiff 基线 + 测 512 时延（加速比分母，延后；需完整SD21+RAM+旧deps）
- ⬜ 4K 分块推理出图（100 张）
- ⏳ JIT 导出 + 自定义 runner（已验证 fp16 trace 无NaN + runner加载OK；待用最终ckpt打包）

## 关键技术决策
- 模型: `models/light_model.py` LightLatentDiff。前向=VAE.encode(LQ)→UNet预测eps→显式x0=(lat-sqrt(1-a)·eps)/sqrt(a)→VAE.decode。t=500（显式公式比 scheduler.step 梯度大1000x，可trace）。
- 训练: `train.py`，fp32，损失=像素L1 + LPIPS(alex) + 潜空间L2，AdamW lr1e-4 cosine。patch256 batch8。
- 退化: `data/degradation.py` 同分辨率(模糊+噪声+JPEG，无下采样)。
- 推理: `infer.py` 512分块高斯融合，fp16 autocast。
- 提交: `export_jit.py`(trace全模型fp16) → `submit/runner.py`(Runner(model_dir).infer)。
- 首次eval(step1000 vs LQ基线4K): 无参全面改善(clipiqa 0.251→0.270, musiq 24.09→26.59, niqe 7.67→6.54, lpips 0.426→0.407)；有参略降(psnr 25.79→24.97，生成式以保真换感知)。

## 服务器关键路径
- 代码: /data/proj/ (models data train.py infer.py eval.py export_jit.py smoke_test.py submit/)
- VAE: /data/weights/sd21base/vae/ (完整SD21/RAM未下，仅教师蒸馏需要)
- OSEDiff: /data/osediff/OSEDiff/ (osediff.pkl, DAPE.pth)
- 数据: /data/datasets/DIV2K_HR/ (548张PNG；提取慢，v2需补)
- 验证集: /data/val/lq/ + /data/val/gt/ (3对4K)
- v1训练: /data/proj/runs/v1/ (unet_step1000.pth... 每1000步存)
- 评测: `bash /data/proj/eval_ckpt.sh <ckpt>` (infer val LQ + eval vs GT)

## SSH 操作要点（避坑）
- 本地无 sshpass，用 paramiko: `ssh_helper.py`(执行命令) / `ssh_cat_upload.py`(流式上传,绕开SFTP)。
- **MSYS 路径转换坑**: 独立参数 `/data/...` 会被 Git Bash 转成 `C:/Program Files/Git/data/...`。调用 ssh_cat_upload 时必须 `export MSYS2_ARG_CONV_EXCL='*'`。ssh_helper 的路径在命令字符串内不受影响。
- **pkill -f 自匹配坑**: pkill -f 'xxx' 会匹配到包含该串的当前命令自身。解决: 用 base64 上传脚本再执行(模式串在base64里不在命令行)，或用 pgrep -P 杀进程树。
- **SFTP 不可用**: 服务器 SFTP 子系统有问题(ENOENT)。一切文件传输走 ssh_cat_upload (exec+cat)。
- 长任务用 nohup 后台 + 轮询日志(/data/*.log)。
- HK 连接偶发 10054 重置，ssh_helper 已加连接重试(6次)。

## 下一步
提交 v3：下载 submit_v3.zip → 重命名为 `赛题二_<作品名>_<队名>_<电话>.zip`（无中文符号）→ 上传赛事网站（每周1次）。上传后看官方分数反馈，再定是否迭代 v4（平衡psnr / OSEDiff蒸馏提质量）。
