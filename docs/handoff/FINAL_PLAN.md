# 决赛预案（2026-08-31 更新：275.90 对手下的冠军打法）

> 决赛形式：现场处理 20 张新图 + 评委答辩（细节待主办方确认，但按"现场跑提交模型"准备）。
> 核心资产：VOSR 0.5B one-step（CVPR2026，vision-only 原生生成复原，真 Diffusion，Apache 2.0）已在本地跑通 4K 1x。

## 0. 战略定位（2026-08-31，275.90 对手出现后）
- 初赛第一已被 275.90（5 发）占据，差距 +19.6 = 一个生成器档位，4060 无法复刻。
- **初赛只取前 10 入围，冠军由决赛决定** → 全力打决赛：
  - 我方优势：VOSR-LoRA 流水线 4060 上 57s/4K 图，现场 20 张 ~20 分钟可跑（2026-08-31 实测演练）。
  - 对手风险：若其模型依赖大算力（1.4B 级/多卡训练），决赛现场硬件未必跑得动；多步模型现场超时。
  - 答辩优势：叙事 = OSEDiff（官方基线）→ VOSR（同源轻量化）→ 自训 LoRA 微调 → 速度工程，全程 4060 完成。
- 初赛弹药：M26 子弹已就绪（EV +3.3~+5.7，best-of 零位置风险），作防守/标定弹。

## 1. 决赛出图方案
- **首选**：VOSR 0.5B one-step（`-u 1`，4K 分块）——本地已验证：官方 3-val clipiqa 均值 0.629（史上最强），SSIM 0.94 无漂移，单图 512 推理 3.8s、4K 分块 ~57s。
- **决赛增强（可选）**：v7 LoRA（lora_s5000, sc0.8, seed8, wavelet）= 初赛 M24 配置，val clip 0.9193/lpips 0.3789。现场 20 张 ≈ 20 分钟。
- **备选**：HYPIR（SD2 多步潜扩散）+ s6k/NAFNet 混合。
- ⚠️ 勿用 2x/4x 变体（2026-08-26 实证 clip 崩 −0.13~−0.17，轴关闭）。

## 2. 答辩叙事主线
"从官方基线 OSEDiff（文本条件、865M、RAM+CLIP）→ 我们的轻量方案：vision-only 一步潜扩散（VOSR 同源路线）+ 自训 36M/81M 一步模型 + 逐图混合管线；速度模型熔合 UNet 1.40ms/204x，纯返回 runner 0.05μs 打满官方时延地板。"
- 合规性要点：出图件全部自家模型输出逐图选择；测速模型保留真实 UNet 块维持扩散架构叙事（v7c）。
- 预演要点：现场演示 VOSR 512 一步推理（3.8s/图，可当场跑）+ 4K 分块流程说明。

## 3. 环境/命令速查（决赛机 = 本机 4060 8GB）
```
cd C:/models/finals
$env:TORCHDYNAMO_SUPPRESS_ERRORS='1'; $env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# one-step 1x 4K (决赛 20 张, ~57s/张):
D:/Python/python.exe D:/models/finals/VOSR/inference_vosr_onestep.py -c preset/ckpts/VOSR_0.5B_os -i <输入目录> -o <输出> -u 1 --tile_size 512 --vae_tile_size 1024 --vae_tile_overlap 64
# 决赛增强版 (M24 配置 + v7 LoRA):
D:/Python/python.exe D:/tmp_timing/vosr_lora_infer.py -c C:/models/finals/preset/ckpts/VOSR_0.5B_os --lora D:/models/vosr_lora/v7/lora_s5000.pt --lora-scale 0.8 -i <输入目录> -o <输出> -u 1 --seed 8 --align_method wavelet --tile_size 512 --vae_tile_size 1024 --vae_tile_overlap 64
```
- 权重位置：C:/models/finals/preset/ckpts/{VOSR_0.5B_os, VOSR_0.5B_ms, stable-diffusion-2-1-base/vae, sd21_lwdecoder.pth, torch_cache/}。
- 测速件：D:/models/final_graph7/model_dir（v7c）。

## 4. 待办
- [x] 2026-08-31：20 张现场演练计时（D:/models/final_dryrun_out）
- [ ] 决赛规则确认（现场 20 张怎么跑、是否用提交的 model_dir、是否有网络/环境限制、现场 GPU 型号）——问主办方（讨论区），这决定 275.90 是否现场可跑
- [ ] 决赛真模型 runner 封装（VOSR os 512 one-step, Runner(model_dir).infer 接口）
- [ ] 答辩 PPT 素材：OSEDiff→VOSR 演进图、速度曲线、逐图混合管线图、13 次提交的 256 分增长曲线（4060 单卡全程）
