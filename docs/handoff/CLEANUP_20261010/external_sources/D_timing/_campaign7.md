

---

## 7. Q2 热点定位与收尾状态（2026-09-15 16:50）

### 7.1 【实测】M24 runner 的 op 级热点（%%BT%%q2_profile.py%%BT%%，6 次 infer 合计）

| 组件 | CUDA 时间 | 折合每次 | 说明 |
|---|---|---|---|
| conv2d（cudnn 813ms + cutlass 407ms + …） | 1.87 s | ~310 ms | **VAE 的活** |
| group_norm | 0.60 s | ~100 ms | 同上 |
| linear / addmm / bmm | 0.80 s | ~133 ms | 0.5B DiT 的活 |
| **to / copy_** | 0.54 s（**5124 次调用**） | **~90 ms，854 次/次** | **fp16↔fp32 搬运，明显浪费** |

**结论：VAE（conv+groupnorm）比 0.5B DiT 更重。** 合法提速手段（规则明文允许 JIT trace / CUDA graph / 不同 patch）：CUDA Graph 固定 512 形状捕获 + 统一 dtype。
**预测**：352.9 → 200~250 ms ⇒ r 0.81→1.2~1.4 ⇒ 合规分 **43.7 → 48~52**（用 %%BT%%score_pred2.py%%BT%% 机械判分）。
**便利条件**：图像可**逐字节不变** ⇒ 同一模型主体零争议；**不占提交次数**即可本地迭代；**不需要重渲 100 张**。

### 7.2 接缝（第二优先）

- test 渲染竖缝 **1.054**（baseline / sc1.2 / adain 都在 1.048~1.055）⇒ 换尺度、换对齐都修不掉；
- 旧世界 **val** 渲染（不同 VAE 分块参数）竖缝 **0.89~1.04，基本无接缝**；
- ⇒【推断】**真凶可能是 VAE 分块**，不是台账旧判断的"latent 随机噪声"；判据：**竖缝→1.00 且 clipiqa 不降**。

### 7.3 本轮踩的工具坑（已写进下一轮提示词 §7）

1. %%BT%%vosr_lora_infer.py%%BT%% 的 %%BT%%torch.hub.load('facebookresearch/dinov2')%%BT%% **要联网做 ref 解析** → 断网 60s 超时崩（重试无效，需离线 wrapper）；
2. %%BT%%measure_dir.py%%BT%% 只找 %%BT%%case{n}.jpg%%BT%% → 对 PNG 渲染必须传 %%BT%%--pattern 'case{n}.png'%%BT%%（否则指标全空、日志 StopIteration）；
3. %%BT%%measure_latency.py%%BT%% v2（逐次 sync）偏悲观，**官方口径更接近 v1 的 200 次均值**。

### 7.4 本轮结束状态

- **机器上无任何任务在运行**（GPU 534 MiB 空闲 / 无 python 进程）。
- %%BT%%upload/%%BT%% 空（无待提交件）；%%BT%%待定_解耦探测/%%BT%% 只剩未用的 %%BT%%Track2_M24img_V9brun.zip%%BT%%。
- 已归档：%%BT%%Track2_TinyV9b——40.5757%%BT%%、%%BT%%Track2_M24img_V7ganrun——94.7249%%BT%%、%%BT%%Track2_M24img_V4grun——105.6401%%BT%%。
- 新增文档：%%BT%%规则细则与判例台账.md%%BT%%（含 §八 分类框架与 fraud 假说）、%%BT%%NEXT_ROUND_PROMPT_20260915PM.md%%BT%%（取代 0916 版）、%%BT%%score_pred2.py%%BT%%（唯一判分工具）。

---
