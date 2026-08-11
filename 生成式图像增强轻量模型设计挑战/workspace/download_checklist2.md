# 下载清单 (2026-08-07, HYPIR-LoRA 微调 E4 需要)

## 1. open_clip convnext_xxlarge 判别器骨干权重 【必须，4.8GB】

HYPIR 的训练判别器 `ImageConvNextDiscriminator` 内部使用 open_clip 的
convnext_xxlarge 骨干 (冻结)。本地未缓存, 实例化时会尝试联网下载。

- **文件**: `open_clip_pytorch_model.bin`
- **大小**: 4.8 GB
- **官方 URL** (HuggingFace):
  `https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/resolve/main/open_clip_pytorch_model.bin`
- **镜像** (国内可用的 hf-mirror):
  `https://hf-mirror.com/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/resolve/main/open_clip_pytorch_model.bin`
- **放置路径**: `D:/models/open_clip/convnext_xxlarge-laion2b_s34b_b82k_augreg_soup.bin`
  (目录不存在就创建)
- 说明: open_clip 2.31.0 的 `create_model_and_transforms("convnext_xxlarge",
  pretrained="laion2b_s34b_b82k_augreg_soup")` 会去 HF 找同名仓库的这个文件;
  若已手动放到 HF 缓存外的本地路径, 我们会把 pretrained 参数改成该本地路径,
  训练时不联网。

## 2. vision-aided-loss (pip 包, 判别器 loss 依赖) 【约 50KB, 若 pip 失败】

- 命令: `/d/Python/python.exe -m pip install vision-aided-loss`
- 若网络不可达, 手动下载 wheel:
  `https://pypi.org/pypi/vision-aided-loss/json` 里找 files[].url
  (通常 `https://files.pythonhosted.org/.../vision_aided_loss-0.1.0-py3-none-any.whl`)
- 放置后: `/d/Python/python.exe -m pip install <wheel路径>`

## 状态跟踪
- [ ] convnext_xxlarge 权重 (4.8GB) -> D:/models/open_clip/
- [ ] vision-aided-loss 安装成功
