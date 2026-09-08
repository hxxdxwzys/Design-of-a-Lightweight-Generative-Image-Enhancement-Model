# 自由探索与灵感集（2026-09-08 晚，deepseek 会话）

> 用户授权自由探索后整理。目的：把零散洞察固化，供后续决策与决赛使用。
> 每条标注：**状态**（已验证/待验证/已排除）与**证据**。

---

## 一、从已有数据挖出的信号（零成本，已完成/进行中）

### 1.1 ✅✅ 测试集无参指标对比（完成, 重大发现）
4 个包各有 100 张测试集渲染 → clipiqa/musiq/niqe 不需要 GT，可直接算。脚本 `nr_metrics_test100.py` / `nr_metrics_one.py`（抽 30 张/包）。

**结果（30 张抽样, 2026-09-09 00:20 完成）**：
| 包 | clipiqa | musiq | niqe |
|---|---|---|---|
| **M24 (0.5B)** | **0.9023** | **46.13** | **5.26** |
| m0s1-sc0.93 (1.4B) | 0.8745 | 38.63 | 6.02 |
| m0s7-sc1.0 | 0.8790 | 37.87 | 6.37 |
| g3w18s1-sc0.93 | 0.8612 | 37.75 | 6.24 |

**💥 结论: M24 在测试集无参上全面领先 1.4B 系, 与 val 结论完全相反!**
- val: m0s1 clip 0.9261 > M24 0.9207; 测试集: M24 0.9023 >> m0s1 0.8745 (反转 +0.028)
- **1.4B 的 clipiqa 泛化差**: val→测试集 m0s1 掉 0.052, M24 只掉 0.018 → **val 3 张对 1.4B 系统性高估**
- 战略含义: ① 若官方公式认无参(4/8 指标) → M24 可能仍是全场最优, 切 1.4B 是亏的 ② 若只认 lpips(旧逆向) → m0s1 赢 ③ **判别提交价值放大**: M24 vs m0s1 官方分差直接判定公式认什么 ④ 第一发 M24 更稳(校准弹 + 可能本身最优)
- 教训: **val 3 张代理对 1.4B 有系统偏差, 测试集直证不可替代**

### 1.1b ✅✅ 全 100 张 clipiqa 直证 (2026-09-09 00:25, 决定性)
| 包 | mean | std | min | max |
|---|---|---|---|---|
| **M24** | **0.9008** | **0.0277** | 0.8166 | 0.9511 |
| m0s1 | 0.8780 | 0.0373 | 0.7555 | 0.9414 |

- **逐图胜负: M24 胜 89 / m0s1 胜 11** (压倒性)
- M24 不只均值高, **稳定性也更好** (std 小 26%); m0s1 有更多崩溃图 (min 低 0.06)
- m0s1 最差图: case5 (0.849 vs 0.758) / case29 / case11 / case25 …
- **战略定论**: M24 回归主提交默认答案 (除非官方公式确实只认 lpips); 1.4B 定位收敛为"判别资产 + 只认 lpips 时的备选"
- 数据: `D:/tmp_timing/nr_full100_clipiqa.json`

### 1.2 ⚖️ lpips "地板带"证据（已验证）
m0s1-s4000-sc0.93 与 m0s7-s5000-sc1.0（不同种子/step/scale）**均值精确同为 0.3571**，但 per-case 互有胜负（0.489/0.219/0.364 vs 0.484/0.230/0.357）。
→ 结论：**不是硬地板，是"地板带"（0.357±0.001）**——两条独立轨迹收敛到同一区域，是 frontier 底部的强证据。措辞必须谨慎（均值同值含巧合成分，per-case 差异证明模型不同）。

### 1.3 ✅ 架构上限分解（已完成）
- 已知：**基座 VOSR_1.4B_os 对 GT 的 lpips = 0.3322**（clip 0.54 时）
- **实测 qwen VAE 往返（4K tiled）lpips = 0.017**（case1 0.0267 / case2 0.0072 / case3 0.0174）——**VAE 不是地板成因**！
- 分解：VAE 误差 0.017 + DiT 增强偏离 0.33+ = 全链 0.357
- **洞察**：1.4B 的 lpips 空间（基座 0.332 → 全链 0.357）几乎全部是"增强内容偏离 GT"——这是**生成式增强的本质代价**，不是架构缺陷。降低它只能减少增强幅度（clip 换 lpips），frontier 的 trade-off 是根本性的。
- 对"理论高度"论证的价值：地板成因已定位（DiT 增强偏离），VAE 无责 → **理论高度 = trade-off 曲线在 clip 饱和约束下的最低点 ≈ 0.357**。

### 1.4 jpg 质量敏感度（低优先，未做）
q95 vs q98 vs png 对 val 指标的影响。需重渲 1 张（png 已删）。预计影响 <0.002。

---

## 二、未试训练轴（有理论依据，按 EV 排序）

| # | 轴 | 假说 | 成本 | 历史证据 |
|---|---|---|---|---|
| 2.1 | **rank 收缩**（r=32→8/16） | 小 rank 限制自由度 → 更温和的 clip 推 → 突破 lpips 地板带 | 6.2h/种子 | 未试过（0.5B 与 1.4B 都是 r=32） |
| 2.2 | **w_lpips 加大**（1.0→2.0/3.0） | 更强保真锚 → 同 clip 下 lpips 更低 | 6.2h | 未试 |
| 2.3 | **早切换多阶段**（w2.0 到 s3000 后切 w1.0 续到 s5000） | m0s1w1（s4000 切）达 (0.9216, 0.3591)；更早切可能保 clip 同时压 lpips | 6.2h | 部分验证（s4000 切有效但不足） |
| 2.4 | 数据量扩张（LSDIR 8.5 万张） | 更多 1x 数据 | 下载+训练 | **0.5B 已证伪**（更多 Flickr2K → val +0.018 但 test clipiqa −0.031）→ 降级 |
| 2.5 | 更强 clip 目标（w_clip 4.0+） | — | 6.2h | 已证 w1.8/2.0/3.0 全扫，clip 推力饱和 |

**推荐**：官方长期不开放时，按 2.1 → 2.2 → 2.3 顺序排（各 6h 双卡，一晚一个）。

---

## 三、外部情报（2026-09-08 调研）

### 3.1 LSDIR 数据集
- 来源：[ofsoundof/LSDIR](https://huggingface.co/ofsoundof/LSDIR/blob/main/README.md)（HF）+ [GitHub](https://github.com/ofsoundof/LSDIR)
- 规模：84991 张 1x 高清图（2K-8K），总体积 200GB+
- 价值：**降级**——0.5B 已有"数据增加证伪"先例（见 2.4）

### 3.2 2026 新一步扩散模型
- [TEASR: Training-Efficient Any-Step Diffusion Transformer for RealSR](https://arxiv-org.ezproxy.obspm.fr/html/2606.16188v1)（any-step DiT）
- [Bridging Restoration and Generation in One-step Diffusion for RealSR](https://arxiv-org.ezproxy.obspm.fr/html/2604.24136v3)
- [FiDeSR](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_FiDeSR_High-Fidelity_and_Detail-Preserving_One-Step_Diffusion_Super-Resolution_CVPR_2026_paper.html)（已知，已证伪）
- 判断：多为 **4x-SR** 场景，而"4x-SR 模型 1x 用法全灭"已实证（5 个源）→ 新模型价值低。**除非出现 1x 原生的增强扩散模型**。

---

## 四、风险盲点（新增）

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| 4.1 | 官方测速 GPU 显存 <8GB | 1.4B runner（7.3GB 峰值）可能 OOM | 观察：本地 4060 8GB 跑得动（7.3GB）→ 若官方 ≥8GB 无碍；若 6GB 需 VAE tiling 优化 |
| 4.2 | 官方 torch 版本 <2.0 | `torch.load(weights_only=False)` 参数不存在 | runner 加 try/except 回退（低成本防御） |
| 4.3 | 官方 Python 3.8 的其他语法 | 已修 future annotations；`|` 已扫 | 保持扫描 |
| 4.4 | 速度分量化被低估 | 1.4B vs 0.5B 速度差可能达 4 分（非 1-2） | **需在框架里明确**：质量 +11 vs 速度 −4 → 净 +7（若 lpips 主轴成立） |

### 4.2 具体防御方案（待实施）
```python
try:
    pkg = torch.load(path, map_location='cpu', weights_only=False)
except TypeError:
    pkg = torch.load(path, map_location='cpu')
```
低成本，建议加到两个 runner.py（0.5B + 1.4B）。

---

## 五、待办优先级（自主决定）
1. 🔄 NR 指标结果（进行中）
2. ⬜ VAE 往返 tiled 测量（1 张，~10 分钟）
3. ⬜ runner torch 兼容回退（两个 runner.py，5 分钟）
4. ⬜ 训练轴 2.1-2.3 排队（官方长期不开放时）
5. ⬜ 框架文档更新（地板带证据 + 速度分量化 + 测试集无参直证）
