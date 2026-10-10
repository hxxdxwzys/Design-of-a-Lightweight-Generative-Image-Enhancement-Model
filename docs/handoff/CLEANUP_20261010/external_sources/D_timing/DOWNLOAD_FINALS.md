# 决赛真 Diffusion 候选 —— 手动下载清单（请用户协助）

> 背景：CODiff 权重挂在 Google Drive（国内不可达，需用户代理手动下载）；
> VOSR 权重在 HF/ModelScope（代理/镜像可下，优先让我自己试，下不动再麻烦用户）。
> 放置路径统一 D:/models/finals/，完成后告知，我来对接 1x 验证。

## 1. CODiff（ICCV2025，一步扩散去 JPEG 伪影，天然 1x）—— 需用户手动下载
- 仓库：https://github.com/jp-guo/CODiff （先 clone 仓库代码）
- 权重文件 1（CODiff 主权重）：
  - URL: https://drive.google.com/file/d/1SHZQs8fu3K419jlENoY8qdTKbjAOL0i7/view
  - 放置：D:/models/finals/codiff/（文件名以下载页显示的为准，勿改名）
- 权重文件 2（CaVE 压缩感知视觉嵌入）：
  - URL: https://drive.google.com/file/d/1SZo5UMSMEcYy4Wr-OIQgTk8hUXiFMIbK/view
  - 放置：D:/models/finals/codiff/
- 配套：SD2.1 基座已在本机（D:/models/sd21base），无需再下。

## 2. VOSR（CVPR2026，vision-only 生成复原，0.5B 一步蒸馏）—— 先让我自行下载
- 仓库：https://github.com/cswry/VOSR
- 权重源 A（HF）：https://huggingface.co/CSWRY/VOSR
- 权重源 B（ModelScope 备选）：https://modelscope.cn/models/LULALULALU/VOSR_CKPT
- 配套 DINOv2 条件编码器（仓库 README 会给依赖；需 torch cache 权重）
- 若 hf-mirror 也断流/失败，再按本清单格式列给用户。

## 备注
- 决赛出图与答辩时间未定，此清单不急；今晚主线仍是 v2b sweep → 140 阶梯混合件。
- 下载完成请告知文件大小与实际文件名，我核对后做 1x 验证（VOSR -u 1 漂移检查为第一优先）。
