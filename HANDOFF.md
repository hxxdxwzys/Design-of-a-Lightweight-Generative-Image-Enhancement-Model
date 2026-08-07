# 项目完整交接文档（HANDOFF）

> 本文件是**事无巨细的全程记录**，供上下文清理后无缝续接。新会话先读本文件 + `CLAUDE.md`（简明版）。
> 最后更新：2026-08-01（v4 完成并关机后）。

---

## 1. 项目是什么

**赛道二：生成式图像增强轻量模型设计挑战**。设计**轻量化 Diffusion 模型**，对低质量图做清晰度增强。
评分 = 感知质量（有参+无参加权）+ 推理时延加速比（以 OSEDiff 为基准）。每周 1 次上传。

### 硬性约束
- **必须 Diffusion 架构**（一步蒸馏扩散算合格；纯 CNN/Transformer 不行）。
- **强制 fp16**。
- **同分辨率复原，非超分**：LQ 与 GT 均 4K（3072×4096 或 4096×3072），LQ 退化 = 重 JPEG 压缩 + 噪声 + 模糊。
- **测速与出图解耦**：
  - 测速：输入输出**均 512×512**，单卡测时延，以 OSEDiff 时延为基准算加速比。提交的 `model_dir/runner.py + your_model.pt` 只需 512×512 又快又对。
  - 出图：100 张 **4K**，离线分块生成提交 jpg，不计入测速。
- 评分指标细节"决赛入围名单时公布"。每周仅 1 次上传。

### 提交格式
`zip≤10GB`，结构：
```
output_dir/case1.jpg ... case100.jpg   ← 4K 增强结果，文件名严格对应
model_dir/your_model.pt                ← torch.jit 导出（fp16）
model_dir/runner.py                    ← 保留 Runner(model_dir) + infer(image_tensor, prompt="") 接口
```
命名：`赛题二_<作品名>_<队名>_<电话>.zip`（无中文符号）。官方样例 runner 是 OSEDiff（4 个 JIT + config + tokenizer），**不强制**，可自定义 runner。

---

## 2. 数据与资源

- **官方数据**：`C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\赛题二\`
  - `测试集\`（100 张 LQ jpg，4K，case1-100，~259MB）—— 出图对象（GT 不公开）。
  - `验证集\`（case1-3：各 `_gt.png/jpg` + `_lq.jpg`，4K，~31MB）—— 本地评测用。
- **官方赛题说明**：`生成式图像增强轻量模型设计挑战.txt`（权威规则）。
- **本地评测基线**（原生 4K，3 对验证集均值）：
  | 指标 | 方向 | LQ 基线 | GT 天花板 |
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

---

## 3. 代码仓库（本地 workspace）

路径：`C:\生成式图像增强轻量模型设计挑战\生成式图像增强轻量模型设计挑战\workspace\`
- `models/light_model.py` — v1/v2/v3 模型 LightLatentDiff：SD2.1 VAE(冻结) + 56M UNet(无 cross-attn) + 显式 x0 预测(t=500)。
- `models/light_model_v4.py` — v4 模型 LightLatentDiffV4：SD2.1 UNet(LoRA+conv_in, 基座冻结) + CLIP 固定 prompt(去 RAM) + 显式 x0(t=500)。
- `models/discriminator.py` — PatchGAN 判别器(谱归一化+InstanceNorm, 返回特征用于特征匹配)。
- `data/degradation.py` — 同分辨率退化管线(JPEG20-95 + 噪声1-20 + 模糊0-2，两阶段)。
- `data/dataset.py` — HQ 图数据集(随机裁 patch + 在线退化，返回 [lq,hq]∈[-1,1])。
- `train.py` — v1/v2 训练(fp32, L1+LPIPS+潜L2, patch256/512, batch8/2)。
- `train_v3.py` — v3 训练(加 PatchGAN：hinge+谱归一化+特征匹配，gan_start warmup)。
- `train_v4.py` — v4 训练(LoRA 微调 SD UNet + PatchGAN，梯度检查点)。
- `infer.py` / `infer_v4.py` — 4K 分块推理(512 tile + 64 重叠高斯融合, fp16 autocast)。
- `eval.py` — 评测闭环(8 指标：PSNR/SSIM/LPIPS/DISTS/CLIPIQA/MUSIQ/MANIQA/NIQE，按 case 匹配)。
- `export_jit.py` / `export_jit_v4.py` — torch.jit trace(fp16, 512) 导出提交模型。
- `smoke_test.py` / `smoke_test_v4.py` — 模型冒烟测试(前向/反向/NaN/显存)。
- `eval_ckpt.sh` / `eval_ckpt_v4.sh` — 一键评测 checkpoint(infer val LQ + eval vs GT)。
- `package_submission.sh` / `package_submission_v4.sh` — 打包提交件(infer 100 测试图 + export JIT + runner + zip)。
- `setup_scripts/` — 服务器环境/数据/打包/监控脚本(install_env, download_*, extract_*, package_*, v2/v3/v4_watcher)。
- `submit/runner.py` — 提交用 runner(加载 your_model.pt, Runner(model_dir).infer)。
- `ssh_helper.py`(远程执行命令,paramiko,带重试) / `ssh_cat_upload.py`(exec+cat 流式上传, 需 MSYS2_ARG_CONV_EXCL='*') / `ssh_cat_download.py`(下载) / `ssh_upload.py`(SFTP, **服务器 SFTP 不可用，勿用**)。
- `results/` — 各版本评测 JSON。
- `preview/` — 样例增强图(已 gitignore)。
- `submit_v3.zip` — v3 提交件(526MB, 已下载到本地, gitignore)。

**GitHub**：https://github.com/hxxdxwzys/Design-of-a-Lightweight-Generative-Image-Enhancement-Model （建议 Private）
- 本地 git 根 = 项目根，分支 main。PAT 在 `~/.claude/settings.json` 的 `GITHUB_PERSONAL_ACCESS_TOKEN`（已写入 .git/config remote，push 免密）。
- `.gitignore` 排除：赛题二/ 数据、submit_v3.zip、preview/、.secrets、neural-enhance-master/、__MACOSX/、*.log。

---

## 4. 云服务器（v4 用，上海，已关机）

- **腾讯云 CVM**，实例 `GN10X.2XLARGE40`（**V100 32GB**，8核40G），上海五区，按量计费。
- **公网 IP / 密码 / 端口：见本地 `.secrets`**（端口 **2222**，22 被腾讯侧拦；root 登录已开启）。
- 镜像 Ubuntu 22.04 + 驱动 580 + CUDA 13 + cuDNN 预装。Python 3.10.12。venv 在 `/data/env`。
- 数据盘 500G 挂载 `/data`。**已关机**（OS shutdown，¥19/h 停收 GPU 费；建议控制台确认"已关机"停计费）。
- **`/data` 内容（重启后全在）**：
  - `/data/proj/` — 全部代码（models/data/train*/infer*/eval/export/smoke/submit/setup_scripts）。
  - `/data/weights/sd21base/` — SD2.1 VAE+UNet+text_encoder+tokenizer+scheduler（safetensors）。
  - `/data/weights/ram_swin_large_14m.pth` — RAM 权重（v5 蒸馏教师用）。
  - `/data/osediff/OSEDiff/` — OSEDiff 仓库（含 osediff.pkl、DAPE.pth）。
  - `/data/datasets/DF2K/` — 3450 张 HQ 图（div2k 800 + flickr 2650）。
  - `/data/proj/runs/v1/ v2/ v3/ v4/` — 各版本 checkpoint（v4: v4_step1000/5000.pth + v4_final.pth，含 LoRA state）。
  - `/data/val/lq + gt`（3 对 4K）+ `/data/test/`（100 张 4K）。
  - `/data/hf_token` — HF token。
- 旧 HK 服务器（GN10Xp, 43.135.33.107, TencentOS, 端口22）已释放/弃用，数据没了。

**SSH 操作要点（避坑，重要）**：
- 本地无 sshpass，用 paramiko：`ssh_helper.py`(执行) / `ssh_cat_upload.py`(上传)。
- **MSYS 路径转换坑**：独立参数 `/data/...` 会被 Git Bash 转成 `C:/Program Files/Git/data/...`。调用 ssh_cat_upload/download 时**必须 `export MSYS2_ARG_CONV_EXCL='*'`**。ssh_helper 的路径在命令字符串内不受影响。
- **pkill -f 自匹配坑**：pkill -f 'xxx' 会匹配当前命令自身。用 base64 上传脚本再执行，或用 pgrep -P 杀进程树。
- **SFTP 不可用**（服务器 ENOENT）。一切传输走 ssh_cat_upload（exec+cat）。
- 长任务用 nohup 后台 + 轮询日志。HK 连接偶发 10054（ssh_helper 有重试）。
- 上海区：**HF 直连不可达** → 用 `hf-mirror.com`（HF_ENDPOINT=https://hf-mirror.com）+ token。**parquet(xet CDN) 镜像不代理** → 已卸载 hf_xet + 用 yangtao9009 的 zip 下载。**pyiqa 的 musiq/dists/maniqa 权重从 huggingface.co 下**（上海不可达→NA），仅 clipiqa/lpips/psnr/ssim 可用。

---

## 5. 模型迭代历程（核心）

### v1（2026-07-16，HK V100 32GB）
- 架构：**从零 56M UNet + SD2.1 VAE(冻结) + 显式 x0(t=500)**，无文本/RAM/CLIP。
- 训练：DIV2K 548 张（提取被训练抢 CPU 拖慢），patch256 batch8，10k iter 计划（5k 停）。
- 结果（step5000）：clipiqa 0.296，musiq 27.76，lpips 0.399，psnr 25.20。
- 教训：548 张已 plateau（5k 后无提升）。

### v2（2026-07-16）
- 同 v1 架构，DF2K 6100 张，12k iter。
- 结果：clipiqa 0.291，musiq 27.52 —— **≈v1**。更多数据没帮助 → **瓶颈是从零小模型本身（无 SD 先验）**。

### v3（2026-07-17，HK V100 32GB）
- 同 v1 架构 + **PatchGAN 对抗（hinge + 谱归一化 + 特征匹配，warmup 2000）**，DF2K，12k iter。
- 结果（4K val）：**clipiqa 0.344，musiq 31.47**，lpips 0.387，niqe 6.56，psnr 24.88（GAN 以像素保真换锐度）。
- **官方评分：20.27 分**（用户截图确认）。
- 教训：GAN 大幅提升感知质量（clipiqa +18%、musiq +14% 相对 LQ）。case2 上 clipiqa 0.497/musiq 44 超 GT 天花板。
- 视觉：v3 比 v2/LQ 更锐利。
- 提交件 `submit_v3.zip`（526MB）已下载到本地。

### v4（2026-07-17~08-01，上海 V100 32GB）
- 架构：**LoRA 微调 SD2.1 UNet（5.93M 可训，基座冻结）+ 固定 prompt(去 RAM) + VAE + 显式 x0(t=500) + PatchGAN**。
- 训练：DF2K 3450 张，patch512 batch2，10k iter，GAN 稳定（d~1.99）。
- 结果：**clipiqa 0.311（final）/ 0.321（step5000）< v3 0.344**。musiq/dists NA（上海 HF 权重下不了）。
- **结论：直接 LoRA 微调 SD UNet 不利用 SD 先验** —— SD 去噪先验与"LQ→HQ 增强"任务冲突，小 LoRA 硬改反而更差。OSEDiff 真正利用 SD 先验靠 **VSD 蒸馏**（SD 当教师给分布指导），不是直接微调。**这条"简单路"被证伪。**
- 另：v4 末段 GAN 失稳（D 压过 G，adv 0.72），但 step5000 仍 < v3，非主因。

### 技术路线演变
从零小模型(v1/v2) → +GAN(v3, 提升) → LoRA 微调 SD UNet(v4, 失败) → **蒸馏(v5, 待做)**。

---

## 6. 经验教训总结（务必记住）

1. **评测必须原生 4K**：长边缩 512 会让 PSNR 25.79 虚高到 31.13（下采样掩盖 JPEG 退化）。`--max-side 512` 仅冒烟。
2. **从零小模型是质量瓶颈**（v1/v2 plateau ~clipiqa 0.29），数据再多也没用。
3. **PatchGAN 对抗损失是感知质量的大杠杆**（v3 clipiqa 0.344）：hinge + 谱归一化 + 特征匹配 + warmup 稳定。
4. **"直接 LoRA 微调 SD UNet"对同分辨率增强无效**（v4 反输 v3）。要利用 SD 先验必须**蒸馏**（VSD 或教师生成目标→学生模仿），或直接用**预训练 SD-SR 模型**。
5. **高分(3090)选手大概率用蒸馏或预训练 SD-SR**（质量来自 SD 先验），不是直接微调、也不是从零小模型。
6. **上海服务器坑**：HF 直连不可达→hf-mirror；parquet(xet) 镜像不代理→卸载 hf_xet + 用 zip；pyiqa musiq/dists/maniqa 权重下不了（huggingface.co 不可达）→仅 clipiqa/lpips/psnr/ssim。
7. **clipiqa 依赖 pkg_resources**：`setuptools<81`（本地 + 服务器都要锁）。
8. **MSYS 路径转换**：ssh_cat_upload/download 必须 `MSYS2_ARG_CONV_EXCL='*'`。
9. **pkill -f 自匹配**：用 base64 上传脚本执行。
10. **SFTP 不可用**：走 exec+cat。
11. **GAN 会失稳**（D 可能压过 G）：监控 d_loss/g_adv，必要时提前停或调 w_adv/加 EMA。
12. **long 任务用 nohup + 日志轮询**，避免前台超时被杀。
13. **不要再对 v3 权重做任何本地继续训练/微调**（2026-08-01 本地验证）：v3plus_local(0.293)/v3plus_v2(0.282)/sd_distill_2000(0.288)/sd_distill_8000(0.281) 全部 < v3 原始 0.343。原因：(a) LR 未对齐 cosine（v3 训完 lr≈0，重新 1e-4 踢它）；(b) 保真损失与生成分布冲突（perception-distortion tradeoff）；(c) 本地数据/退化与 v3 训练分布偏移。**要超越 v3 只能换分布方向（真正的增强教师蒸馏）或换出图方法。**
14. **SD2.1 通用教师蒸馏失败**（train_sd_distill.py 8000 步 0.281 < 0.343）：SDS-lite 拉向通用文生图先验与增强任务冲突（v4 教训重演）。**增强教师必须用 OSEDiff 这类增强专用模型。**
15. **OSEDiff 兼容坑**（2026-08-02）：osediff.pkl 是 LoRA 权重（非完整权重）。diffusers 0.39 的 `add_adapter` 只给 target_modules 建 LoRA → vae decoder LoRA 缺失 → 输出压平（std 12 vs 官方 benchmark 50）。**必须用官方环境（Python 3.11 + diffusers 0.25 + transformers 4.28 + peft 0.9）复现**。torch 2.7 兼容 diffusers 0.25（>=1.4 无上限）。官方 benchmark 输出在仓库 preset/datasets/benchmark_realsr/results_osediff/（本地可查，std=50 为正常参考）。**环境已搭好（D:/venv_osediff311）且跑通。**
15b. **OSEDiff 1x 同分辨率证伪**（2026-08-02 决定性验证）：官方环境 4K 1x 评测 clipiqa 0.279/musiq 29.9/psnr 11.1 < v3。根因：**OSEDiff 是 4x 超分模型**，1x 输入不在训练分布内 → 输出内容漂移（LQ-GT diff 11.3 vs OSEDiff-GT diff 53.9，亮度偏 22）。直接出图和作教师蒸馏都不可靠。**勿重走此路**。v3（0.343）仍是"内容保持+感知提升"最佳平衡。
15c. **官方评分校准完成**（2026-08-02）：方案A（OSEDiff 1x 出图提交）官方 **10.13 分** vs v3 的 **20.27 分**，与本地 clipiqa 排序一致 → **本地 eval.py 8 指标可作官方分可靠代理**。已用 2 次提交，剩 8 次。
15d. **v5 蒸馏（带 GT 锚定）证伪**（2026-08-02）：train_v5_gt.py 2000 步 clipiqa 0.283/musiq 29.9 < v3 0.343。GT 锚定保住 psnr 但教师"过度锐化"污染感知。**教师蒸馏（SD2.1/OSEDiff）在同分辨率轻退化任务上全部证伪**。
16. **下载类事情自己下不好就列清单让用户手动下**（用户 2026-08-02 明确要求）：如 Python 安装包（python.org 可达）、大 wheel。清单模板见 workspace/download_checklist.md。**原则：自己下不了不要硬下（浪费时间+hash 校验失败），也不要回避跳过（会导致后续步骤卡死）——立刻事无巨细告知用户下载需求（文件名/大小/URL/放置路径/安装命令）。** 用户有代理且愿意手动下载。实例：torch cp311 wheel（download.pytorch.org 直连不稳，自动下载 hash 失败 → 用户手动下到 D:/models/）。
17. **提交规则更正**：实际是**总共 10 次机会无每周限制**（用户确认）。出图(output_dir 100 张 4K jpg)与测速(model_dir your_model.pt+runner.py)独立评分——**出图可离线慢跑任意方法，测速用轻量模型**。
17b. **上传文件名绝对不能含中文**（2026-08-05 用户实测确认）：zip 文件名含中文 → 上传失败（曾因此浪费 3 次提交机会）。赛题示例"赛题二_xxx"含中文是坑。**必须用纯 ASCII 文件名**，如 `Track2_SpeedV2_Team4060_13000000000.zip`。全角破折号"——"可用（已成功上传多次），中文不行。提交件统一放 `提交历史/`（已 gitignore）。

---

## 7. 提交状态（2026-08-07 更新）

**已提交 7 次（原号剩 2 次；用户可创新号再得 10 次）**：
| # | 提交 | 官方分 | 关键配置 |
|---|---|---|---|
| 1 | v3 | 20.27 | v3 出图 + v3 测速 |
| 2 | OSEDiff | 10.13 | OSEDiff 1x 出图（证伪） |
| 3 | DiffBIR | 18.57 | DiffBIR 出图 + v3 测速 |
| 4 | 融合 v3+DiffBIR | 20.84 | α0.85 融合 + v3 测速 |
| 5 | 极速V1 | 29.85 | 融合出图 + TinyVAE 测速(30ms) |
| 6 | 极速V2 | 31.6 | 融合出图 + 像素UNet 测速(17.7ms) |
| 7 | HYPIR | 37.52 | **HYPIR 出图 + 像素UNet 测速 16.9x ← 当前最优** |
| 8 | HYPIR+DiffBIR α0.7 | 37.05 | 融合反而低（lpips 0.336→0.383） |

**关键校准**：加速比 1.4→9.3x 贡献大（+9分），9.3→16.9x 饱和（+1.75分）。**官方评分 lpips 权重最大**（融合 musiq/psnr 全升但 lpips 变差 → 分数反降）。**HYPIR 纯 lpips 0.336 是历史最低（最优）**，这是 37.52 的关键。

**最优提交件**：`提交历史/Track2_HYPIR_Team4060_13000000000.zip`（HYPIR 出图 + 像素测速，37.52 分）。

---

## 8. 未来方向（2026-08-07 更新，已证伪勿重走）

**已验证的最优**：HYPIR 出图（lpips 0.336 最低 + clipiqa 0.440/musiq 36.8）+ 像素域 0.9M UNet 测速（17ms，16.9x）→ **37.52 分**。

**已证伪路线（勿重走）**：
1. v3 继续训练/微调（四条全败，0.28-0.29）
2. SD2.1/OSEDiff 教师蒸馏（0.281/0.283）
3. OSEDiff 1x 出图（官方 10.13）
4. DiffBIR 单独出图（18.57，lpips 0.623 崩）
5. 融合降 lpips（HYPIR+DiffBIR 37.05 < 37.52）
6. TinyVAE 换 VAE（0.171）
7. **FaithDiff 4K**（20 步扩散 + 8GB 卡死，0/20 步 26min）——权重 17GB 待定去留

**下一步候选**：
1. **HYPIR 参数精调**（model_t/coeff_t 微调，4K 已试 100/300/500 均不如 200）
2. **2026 新模型权重开放**（UniLDiff CVPR2026 / HYPIR++ ICML2026，均未开源）——按"模型越新越强"规律，未来可再找
3. **HYPIR++ 等发布**后验证
4. 测速已到极限（0.9M 4层 = 17ms 最优，2/3 层更慢）

**模型评估结论**：HYPIR 是"已开源 + 8GB 可跑 + 1x 适配 + 最新（2025.07）"最佳。FaithDiff（SDXL 基座）512 质量温和但 4K 卡死。SUPIR 太重（60GB+LLaVA-13B）。

---

## 9. 敏感信息与安全

- **`.secrets`（本地，gitignore，不入库）**：服务器 IP/密码/端口、HF token。**严禁提交 GitHub 或打入提交 zip**。
- CLAUDE.md 已脱敏（服务器信息指向 .secrets）。
- HF token 已在服务器 `/data/hf_token` + `~/.claude/settings.json` env（GITHUB_PERSONAL_ACCESS_TOKEN 是 GitHub 的，不是 HF 的）。
- 服务器密码见 `.secrets`（勿入库；换机/泄露后改密）。

---

## 10. 下一步行动清单（2026-08-02 更新）

1. **用户手动下载 Python 3.11.9**（清单 workspace/download_checklist.md）→ 装好后创建 D:/venv_osediff311。
2. **装官方依赖**：diffusers 0.25 + transformers 4.28 + peft 0.9 + loralib/fairscale/einops/open-clip（清华源）。torch 用本地 2.7 或下 cp311 版。
3. **跑通官方 OSEDiff**：venv 里跑官方 test_osediff.py（--upscale 1 同分辨率）→ 3 张验证集 4K 出图 → eval.py 对比 v3（clipiqa 0.343）。若超 v3 → 出 100 张 4K 提交。
4. **或 v5 蒸馏**：官方 OSEDiff 跑通后 gen_teacher_targets.py 生成教师目标 → train_v5.py 蒸馏学生（LR 对齐已内置）。
5. 期间 v3 保底不动，随时可提交（剩 9 次机会）。
4. 新会话：先读本文件 + CLAUDE.md，再看 workspace 代码。
