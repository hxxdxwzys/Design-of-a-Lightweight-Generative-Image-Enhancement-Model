# 决赛"真 Diffusion"候选调研报告（2025–2026 开源/公开权重，1x 复原增强优先）

> 调研时间：2026-08（模拟）。范围：可用于**同分辨率（1x）图像复原/增强**的 Diffusion（含一步/少步蒸馏），以及团队明确要求核实的 HYPIR++、UniLDiff。
> 结论要点：**HYPIR++ 未开源**（"最先进版"仅在 suppixel 商业化）；**UniLDiff 未开源**（仅 CVPR2026 论文）；新增最有价值的是 **VOSR（CVPR2026，视觉-only 原生生成复原，含一步蒸馏权重）** 与 **CODiff（ICCV2025，一步扩散去 JPEG 伪影，天然 1x）**。

---

## 一、候选清单总表

| 模型 | 年份 | 架构类型 | 权重 URL | 参数规模（估） | 8GB 分块 4K | 1x 适配性 | 风险 |
|---|---|---|---|---|---|---|---|
| **HYPIR**（已持有） | 2025 SIGGRAPH | 多步潜扩散复原（SD2.1 LoRA + 得分先验引导） | [GitHub](https://github.com/XPixelGroup/HYPIR) / [HF SD2.1-LoRA](https://huggingface.co/EricRollei/SAI-SD2.1-for-HYPIR) | SD2.1 UNet ≈865M + LoRA | ✅ 已跑通 | ✅ 天然 1x | 非商用研究许可；多步推理较慢（出图不计时，可接受） |
| **VOSR** | 2026 CVPR | 视觉-only 原生生成复原（0.5B/1.4B，多步 + 一步蒸馏两套） | [GitHub](https://github.com/cswry/VOSR) / [HF](https://huggingface.co/CSWRY/VOSR) / [ModelScope](https://modelscope.cn/models/LULALULALU/VOSR_CKPT) | 0.5B / 1.4B | ✅ 0.5B 极稳；1.4B 需 512 tile | ⚠️ 默认 4x，1x 需验证（框架称"native restoration"） | 1x 行为未在论文验证；DINOv2 条件依赖 |
| **CODiff** | 2025 ICCV | 一步扩散 + 压缩感知嵌入（JPEG 去伪影） | [GitHub](https://github.com/jp-guo/CODiff)（权重 Google Drive） | SD2.1 + LoRA + CaVE（小） | ✅ 512 tile | ✅ 天然 1x（JPEG 专用） | 仅去 JPEG，不处理模糊/噪声；权重在 Google Drive（国内需手动下） |
| **UniLDiff** | 2026 CVPR | All-in-One 扩散复原（1x 去噪/去雨/去雾/去模糊/低光） | 未开源（仅 [论文](https://arxiv.org/pdf/2507.23685) / [CVPR 页面](https://cvpr.thecvf.com/virtual/2026/poster/38761)） | 未公开 | 未知 | ✅ 天然 1x | **未开源/未确认**，不可用 |
| **HYPIR++** | — | — | 未开源（[HYPIR 项目页](https://hypir.xpixel.group/)说明"最先进模型"在 suppixel 商用） | — | — | — | **不存在独立开源版** |
| **SinSR** | 2024 CVPR | 一步蒸馏扩散 SR（ResShift 蒸馏） | [GitHub](https://github.com/wyf0912/SinSR)（权重随仓库/Drive） | ≈119M（ResShift 骨架） | ✅ | ⚠️ 4x（1x 会漂移） | 权重下载链路需核；4x 专用 |
| **ResShift** | 2023 NeurIPS/2024 TPAMI | 少步扩散 SR（残差移位，可蒸馏到 1 步） | [GitHub](https://github.com/zsyOAOA/ResShift) | ≈119M | ✅ | ⚠️ 4x（有 1x 变体） | 4x 为主；1x 需另训 |
| **OSEDiff**（官方基线） | 2024 NeurIPS | 一步潜扩散 SR（SD2.1 LoRA + RAM/CLIP） | [GitHub](https://github.com/cswry/OSEDiff) | SD2.1 + LoRA ≈865M+ | ✅ | ❌ 1x 已证伪（内容漂移） | 团队已证伪，勿重走 |
| **AdcSR** | 2025 CVPR | 一步扩散压缩（去 VAE-enc/文本/交叉注意力，剪枝 UNet/decoder） | [GitHub](https://github.com/Guaishou74851/AdcSR) / [HF](https://huggingface.co/Guaishou74851/AdcSR) | ≈OSEDiff 的 26%（~220M） | ✅ | ⚠️ 4x | 4x 专用；速度叙事优秀 |
| **PiSA-SR** | 2025 CVPR | 一步扩散 + 双 LoRA（像素/语义可调） | [GitHub](https://github.com/csslc/PiSA-SR) | SD2.1 + 2×LoRA | ✅ | ⚠️ 4x（有 AIGC 增强模式） | 可调 λ 控制去噪/压缩伪影，与本退化契合，但 4x |
| **TVT** | 2025 ICCV | 一步扩散 + 重训 VAE（保细结构） | [GitHub](https://github.com/Joyies/TVT) / [HF](https://huggingface.co/Joypop/TVTSR) | SD2.1 UNet + 新 VAE | ✅（自带 tile 推理） | ⚠️ 4x | 4x；`--time_step 1` 一步 |
| **TSD-SR** | 2025 CVPR | 一步扩散（SD3-medium LoRA + 目标得分蒸馏） | [GitHub](https://github.com/Microtreei/TSD-SR)（权重 Drive/OneDrive） | SD3-medium ≈2B + LoRA | ⚠️ 512 tile 偏重 | ⚠️ 4x | 2B 骨干 8GB 偏紧 |
| **TADSR** | 2026 CVPR | 时间感知一步扩散 SR | [GitHub](https://github.com/zty557/TADSR) / [HF](https://huggingface.co/zty557/TADSR) | 未详（SD-based） | 待验 | ⚠️ 4x | 许可 Pi-Lab 1.0（非商用） |
| **FluxSR** | 2025 ICML | 一步流轨迹蒸馏 SR（FLUX 骨干） | [GitHub](https://github.com/JianzeLi-114/FluxSR) | FLUX 骨干（≥3.5B） | ❌/⚠️ 8GB 偏紧 | ⚠️ 4x | 骨干过重，4060 吃力 |
| **AddSR** | 2024/PR | 对抗扩散蒸馏加速盲 SR | [GitHub](https://github.com/NJU-PCALab/AddSR) | SD-based | ✅ | ⚠️ 4x | 4x |
| **DoSSR** | 2024 NeurIPS | 域移位 SDE 驯服扩散先验 SR | [GitHub](https://github.com/AMD-AGI/DoSSR) | SD2.1-based | ✅ | ⚠️ 4x | 4x |
| **DiT4SR** | 2025 ICCV | DiT 真实 SR | [GitHub](https://github.com/Adam-duan/DiT4SR) | DiT（较重） | ⚠️ | ⚠️ 4x | 重 |
| **ODTSR** | 2026 CVPR | 一步扩散 Transformer 可控 SR | [GitHub](https://github.com/RedMediaTech/ODTSR) | DiT | ⚠️ | ⚠️ 4x | 重 |
| **BlurDM** | 2025 NeurIPS | 模糊扩散去模糊（1x） | [论文](https://mlanthology.org/neurips/2025/he2025neurips-blurdm/) | 未详 | 待验 | ✅ 1x（仅去模糊） | 单任务 |
| **One-Step Motion-Deblur** | 2025 | 一步扩散去运动模糊（1x） | [论文](https://huggingface.co/papers/2503.06537) | 未详 | 待验 | ✅ 1x（仅去模糊） | 单任务；权重未确认 |
| **OSDFace** | 2025 CVPR | 一步扩散人脸复原（1x） | [GitHub](https://github.com/jkwang28/OSDFace) | SD-based LoRA | ✅ | ✅ 1x（仅人脸） | 人脸专用，泛图不适用 |

> 说明：参数规模为公开信息估算（SD2.1 UNet≈865M、SD3-medium≈2B、ResShift≈119M）；"✅ 8GB"指 512×512 分块 + fp16 可跑 4K（与团队现有 HYPIR/OSEDiff 分块实践一致）。1x 适配性"⚠️ 4x"指该模型原生为 4x 超分，直接 1x 使用存在与 OSEDiff 相同的内容漂移风险，需验证或微调。

---

## 二、值得深挖的候选（逐段分析）

### 1. VOSR（CVPR 2026，PolyU/OPPO，OSEDiff 同源）—— **首选新增方向**
- **为什么适合**：官方定位是"**视觉-only 原生生成式图像复原框架**"（vision-only，砍掉了 OSEDiff 的 RAM+CLIP 文本路径，改用 DINOv2 视觉条件），这正是团队"去文本路径"路线的权威佐证。同时官方**同时放出多步（25 步）与一步蒸馏（1 步）两套权重**，0.5B/1.4B 两档，HF 与 ModelScope 双源可下、权重齐全。0.5B 模型 fp16 约 1GB，8GB 卡 512 分块跑 4K 毫无压力。
- **答辩叙事**：可以直接讲"从 OSEDiff（官方基线，文本依赖、865M）→ VOSR（视觉-only、0.5B、一步蒸馏）"，技术演进完整、是官方基线作者自己走出的轻量化路线，评委认可度高。
- **风险/待办**：README 默认 `-u 4`（4x 超分），**1x 行为未在论文明确验证**；需本地跑一次 1x（或 `-u 1`）确认无内容漂移。其次依赖 DINOv2 条件编码（需额外下 torch_cache/DINOv2 权重，仓库已给）。
- 来源：[GitHub](https://github.com/cswry/VOSR)、[HF 权重](https://huggingface.co/CSWRY/VOSR)、[ModelScope 权重](https://modelscope.cn/models/LULALULALU/VOSR_CKPT)、[arXiv](https://arxiv.org/pdf/2604.03225)。

### 2. CODiff（ICCV 2025）—— **一步扩散 + JPEG 去伪影（天然 1x）**
- **为什么适合**：本赛道退化是"**重 JPEG 压缩 + 噪声 + 模糊**"，JPEG 伪影是最主要、最影响 LPIPS/CLIPIQA 的成分。CODiff 是**专门针对 JPEG 伪影去除的一步扩散模型**，天然 1x（去伪影不改变分辨率），架构 = SD2.1 + LoRA + 压缩感知视觉嵌入（CaVE），对 QF=1/5/10 的严重压缩效果明确优于同类。作为"一步潜扩散"完全符合赛道定义，且规模小、8GB 分块可跑 4K。
- **答辩叙事**：可讲"针对赛道核心退化（重 JPEG）定制的压缩感知一步扩散，CaVE 显式编码 JPEG 压缩先验引导去噪"——精准扣题、差异化明显。
- **风险/待办**：只处理 JPEG，不覆盖模糊/噪声 → 更适合作为 **HYPIR/VOSR 的辅助后处理或逐图混合组件**，而非唯一模型。权重在 Google Drive（[CODiff](https://drive.google.com/file/d/1SHZQs8fu3K419jlENoY8qdTKbjAOL0i7/view) / [CaVE](https://drive.google.com/file/d/1SZo5UMSMEcYy4Wr-OIQgTk8hUXiFMIbK/view)），国内需用户手动下载。
- 来源：[GitHub](https://github.com/jp-guo/CODiff)、[arXiv](https://arxiv.org/pdf/2502.09873)。

### 3. HYPIR（SIGGRAPH 2025，已持有）—— **继续作为"真多步 Diffusion"主模型**
- **为什么适合**：这是团队当前**最强真 Diffusion 1x 出图**（4K lpips≈0.336、clipiqa≈0.44），本身就是货真价实的 SD2.1 潜扩散复原（得分先验引导），完全站得住"我们用的是 Diffusion"的答辩。SD2.1 LoRA 权重在 HF 公开，8GB 分块已跑通。
- **答辩叙事**：HYPIR 用"扩散产出的得分先验"引导复原，是 SIGGRAPH2025 顶会工作；团队持有权重并在 4K 分块落地，可直接作为决赛主出图方案，无需再冒新模型质量风险。
- **风险/待办**：非商用研究许可（需核对 LICENSE 是否允许赛事提交）；多步推理慢（但出图离线不计时，无碍）。
- 来源：[GitHub](https://github.com/XPixelGroup/HYPIR)、[HF](https://huggingface.co/EricRollei/SAI-SD2.1-for-HYPIR)、[arXiv](https://arxiv.org/abs/2507.20590)、[项目页](https://hypir.xpixel.group/)。

### 4. AdcSR / TVT / PiSA-SR（OSEDiff 家族的一步轻量化，方法学参考 + 潜在速度组件）
- **为什么适合（方法论/叙事）**：三者都是"把 OSEDiff 轻量化/提速"的官方同源工作：AdcSR 砍 VAE-enc/文本/交叉注意力并剪枝（**参数 -74%、9.3× 提速、0.03s@128→512**），TVT 重训 VAE 保细结构且**自带 tile 低显存推理**、`--time_step 1` 一步，PiSA-SR 用双 LoRA 可调"去噪/去压缩伪影 vs 语义细节"。这直接给团队的"轻量一步潜扩散"路线提供权威背书与可复用的工程做法。
- **风险/待办**：均为 **4x 超分**，1x 直接使用有漂移风险（同 OSEDiff），不建议直接出 1x 图；价值在于**方法参考**与答辩时的技术演进叙事。AdcSR/TVT 权重在 HF 公开，可低成本复现其压缩/VAE 思路到 1x。
- 来源：[AdcSR](https://github.com/Guaishou74851/AdcSR)、[TVT](https://github.com/Joyies/TVT)、[PiSA-SR](https://github.com/csslc/PiSA-SR)。

### 5. UniLDiff（CVPR 2026）—— **待观察，暂不可用**
- **为什么关注**：All-in-One 扩散复原（1x 去噪/去雨/去雾/去模糊/低光），与本赛道 1x 增强方向最契合，且是 XPixel（HYPIR/DiffBIR 同门）工作。
- **现状**：截至调研**未找到官方代码/权重**（仅 CVPR2026 论文与 Open Access 页面），与用户"当时未开源"一致。**结论：未开源/未确认，不可作为决赛依赖**，但可追踪后续发布。
- 来源：[arXiv](https://arxiv.org/pdf/2507.23685)、[CVPR 页面](https://cvpr.thecvf.com/virtual/2026/poster/38761)、[OpenAccess](https://www.openaccess.thecvf.com/content/CVPR2026/html/Cheng_UniLDiff_Unlocking_the_Power_of_Diffusion_Priors_for_All-in-One_Image_CVPR_2026_paper.html)。

### 6. HYPIR++ —— **确认：不存在独立开源版**
- 官方 HYPIR README 明确："当前开源版基于 SD2.1；**我们最先进的模型已上线 suppixel.ai / suppixel.cn（商用，未开源）**"。团队所说的"HYPIR++"大概率指这个商用最先进版。**结论：未开源，不可用**；开源可用上限就是已持有的 HYPIR-SD2。
- 来源：[HYPIR README](https://github.com/XPixelGroup/HYPIR)、[suppixel](https://suppixel.ai)。

---

## 三、明确推荐（决赛"真 Diffusion"方案）

### 首选：**HYPIR（已持有）为主出图 + VOSR-0.5B 一步蒸馏 为轻量/答辩主线**
1. **出图（决定感知分）**：继续用 **HYPIR-SD2** 4K 分块出图——它是当前最强、最稳的真 Diffusion 1x 输出，已本地验证 clipiqa 0.44，零额外风险。
2. **答辩主线**：主打 **VOSR-0.5B one-step（CVPR2026）** 作为"轻量一步潜扩散、视觉-only（去 RAM/CLIP 文本路径）"的代表作——与官方基线 OSEDiff 同源、技术路线完全对齐团队既定方向，且官方权重 HF/ModelScope 齐全、可当场跑 512 演示一步生成。**这是评委最难质疑"不像 Diffusion / 不是轻量"的选项。**
3. **加分组件**：**CODiff（ICCV2025，一步扩散去 JPEG）** 针对赛道核心退化做后处理/逐图混合，进一步提升 LPIPS/CLIPIQA。
4. **落地动作**：先本地验证 VOSR `-u 1`（或 1x 输入）是否无漂移；同时列 CODiff 权重 Google Drive 给用户手动下载。

### 备选（若 VOSR 1x 漂移或时间不足）
- **保底 = 纯 HYPIR**：已持有、已验证、货真价实 Diffusion，答辩足够。
- **第二备选 = HYPIR + CODiff 混合**：不引入新 SR 模型，仅叠加 1x 一步去 JPEG 扩散组件，风险最低、叙事清晰。

### 明确不建议
- **不要**再把任何 **4x SR 一步模型（SinSR/ResShift/TSD-SR/TADSR/FluxSR/AdcSR/TVT/PiSA-SR/ODTSR）直接用于 1x 出图**——与 OSEDiff 1x 漂移同因，大概率内容漂移、丢感知分；它们仅作方法学与答辩背书。
- **不要**把 UniLDiff、HYPIR++ 写进决赛方案——均未开源，无法落地。

---

## 四、附：全部来源 URL

**核心候选（权重/代码）**
- HYPIR：https://github.com/XPixelGroup/HYPIR ；https://hypir.xpixel.group/ ；https://huggingface.co/EricRollei/SAI-SD2.1-for-HYPIR ；https://arxiv.org/abs/2507.20590
- VOSR：https://github.com/cswry/VOSR ；https://huggingface.co/CSWRY/VOSR ；https://modelscope.cn/models/LULALULALU/VOSR_CKPT ；https://arxiv.org/pdf/2604.03225
- CODiff：https://github.com/jp-guo/CODiff ；https://arxiv.org/pdf/2502.09873 ；https://huggingface.co/papers/2502.09873
- UniLDiff：https://arxiv.org/pdf/2507.23685 ；https://cvpr.thecvf.com/virtual/2026/poster/38761 ；https://www.openaccess.thecvf.com/content/CVPR2026/html/Cheng_UniLDiff_Unlocking_the_Power_of_Diffusion_Priors_for_All-in-One_Image_CVPR_2026_paper.html
- OSEDiff：https://github.com/cswry/OSEDiff ；https://arxiv.org/pdf/2406.08177
- SinSR：https://github.com/wyf0912/SinSR ；https://arxiv.org/pdf/2311.14760
- ResShift：https://github.com/zsyOAOA/ResShift ；https://huggingface.co/papers/2307.12348
- AdcSR：https://github.com/Guaishou74851/AdcSR ；https://huggingface.co/Guaishou74851/AdcSR ；https://arxiv.org/abs/2411.13383
- PiSA-SR：https://github.com/csslc/PiSA-SR ；https://arxiv.org/pdf/2412.03017
- TVT：https://github.com/Joyies/TVT ；https://huggingface.co/Joypop/TVTSR ；https://arxiv.org/pdf/2507.20291
- TSD-SR：https://github.com/Microtreei/TSD-SR ；https://arxiv.org/abs/2411.18263
- TADSR：https://github.com/zty557/TADSR ；https://huggingface.co/zty557/TADSR ；https://arxiv.org/abs/2508.16557
- FluxSR：https://github.com/JianzeLi-114/FluxSR ；https://huggingface.co/papers/2502.01993
- AddSR：https://github.com/NJU-PCALab/AddSR
- DoSSR：https://github.com/AMD-AGI/DoSSR
- DiT4SR：https://github.com/Adam-duan/DiT4SR
- OSDFace：https://github.com/jkwang28/OSDFace
- ODTSR：https://github.com/RedMediaTech/ODTSR
- CCSR / CCSRv2：https://github.com/csslc/CCSR
- DiffBIR：https://github.com/XPixelGroup/DiffBIR
- BlurDM（NeurIPS2025）：https://mlanthology.org/neurips/2025/he2025neurips-blurdm/
- One-Step Motion-Deblur：https://huggingface.co/papers/2503.06537 ；https://arxiv.org/html/2503.06537
- One-Step Residual Shifting Diffusion（ResShift 一步蒸馏）：https://arxiv.org/pdf/2503.13358 ；https://icml.cc/virtual/2026/poster/66455

**索引/综述（便于追踪后续发布）**
- Awesome-Diffusion-Models-in-Low-Level-Vision：https://github.com/ChunmingHe/awesome-diffusion-models-in-low-level-vision
- Awesome-One-Step-Generation：https://github.com/Luciferbobo/Awesome-One-Step-Generation
- XPixel X-Image-Processing：https://github.com/XPixelGroup/X-Image-Processing
- CVPR2026 底层视觉论文汇总：https://blog.csdn.net/Kobaayyy/article/details/161633934
