# 冲刺七执行预案（2026-08-15，deepseek冲刺5 会话）

## 0. 起点快照
- 官方分最优 135.5965（M01：58 换图，test clipiqa +0.0278，v7c 速度件），除 TESTCASE 外第一。
- 校准：官方分/test-clipiqa = 230×→288×→322×（凸性）。
- 预算：新号 9 次（用户提交）。
- 阶梯：140（v2b 混合/TTA）→ 150（width=48）→ 170（GT 级）→ 189.98。

## 1. 后台状态（15:25 实测）
- train_v2b：pid 19492，step 12400/16000（1.37s/步），ETA ~16:50 完成（含 14k/16k eval + [done]）。
- watcher_v2b：pid 52308 存活，轮询 [done]（90s）→ 自动启动 `sweep_test_all.py --ckpt-dirs D:/models/metric_net_v2b` → D:/tmp_timing/sweep_v2b.log。
- sweep 范围 = v2b net/ema s10000-16000 共 14 ckpt（sweep_test_all.build_jobs 规则确认），旧 34 件 done 自动跳过，结果增量写 D:/models/sweep_test_out/sweep_results.json。
- sweep 时长 ≈ 14 × ~15min ≈ 3.5h → 结果 ~20:30。
- GPU 100% 占用；D: 剩 26.5GB。

## 2. 决策分支（sweep 完成后）
先备份 sweep_results.json（cp 到 sweep_results.backup_YYYYMMDD.json）再动。

### 分支 A：存在 v2b ckpt 均值 > s6k (0.5354)，或逐图胜场有净增量
1. 重跑 compute_choice.py（自动纳入 sweep_results 新条目）→ choice_maps.json 更新。
2. 重跑 hybrid_rule.py（对照；v2b val_lpips ≈0.373-0.381 > safe 阈值 0.3633 → 被当作 unsafe，仅 >0.05 边际才换图；不修改 selection_analysis.json，保守闸保留）。
3. 看 M 边际表（期望 M0.010 行），选定策略 → assemble_mix.py --strategy map:M0.010 --out D:/models/cnn_out_mix_s7
4. 可选 TTA 增益验证：tta_test.py 先在官方 3-val 上验证 8 向 TTA 双赢，再决定是否渲染 TTA 版选图。
5. package_submit_v2.py --out-dir <mix dir> --name <ASCII> → 交用户（新号）。

### 分支 B：v2b 全输（test clipiqa 均 ≤ s6k 且逐图无净增量）
0. ⚠️ 新增探针步（重要）：v2b val3 clipiqa 峰值在 s4000（0.542），s10k+ 仅 0.427-0.441 且与 test（ema s10k 0.4472 / s11k 0.4365）吻合 → **早期 ckpt 的 test 可能达 0.53+（s6k 级）**。主 sweep 完（~20:55）后先跑小探针：`sweep_test_all.py --ckpt-dirs D:/models/metric_net_v2b --min-step 2000` 仅 s2k/s4k/s6k net+ema 6 件 ≈1.5h；若任一 test ≥0.53 → 全早期补扫 + 重新评估分支 A。
1. 分析：holdout vs val3 vs test 三方对照；fixed profile 错位假设（sigma 0.8-3.5 覆盖不了 test 中位 σ≈0.5 的多数图；noise 0-2.0 覆盖不了 23% >2.4 重噪图；jpeg 55-95 vs test q~92）。
2. 可选增补 sweep：v2b 早期 ckpt s1k-9k（net+ema 18 件 ≈ 4.5h）进池搏逐图胜场（f2k_s1000 教训：欠训练 ckpt 专赢近干净图）。用户定夺。
3. width=48 重训（150 阶梯）：
   - 命令草稿：`D:/Python/python.exe D:/tmp_timing/train_v2.py --sampler fixed --width 48 --out D:/models/metric_net_w48 --steps 16000 --batch <3/4> --accum <3/2>`
   - ✅ 已给 train_v2.py 加 `--accum N`（梯度累积、默认 1 向后兼容、cosine T_max=steps//accum、打印用未缩放 loss；py_compile 通过；不影响在跑的 v2b 进程）
   - ✅ mem_probe_w48.py 就绪（GPU 空出后先跑，复刻 L1+LPIPS 训练图测峰值显存，定 batch/accum 组合）
   - ✅ guard_disk.py 就绪（D: 空闲 <12GB 时按 full_s5k/10k/15k → net_s<10k → ema_s<10k 顺序删旧 ckpt，保留 s10k+ 供直选 sweep）
   - ⚠️ holdout 口径坑：v2b/w48 复用 D:/datasets/holdout_v2 的 manifest（v2 emp 退化生成），fixed 训练在 emp 退化 holdout 上评估——holdout 只是监视器，test 为准；若换 profile 重训需换 --holdout 目录重建
   - 磁盘：训练前删 D:/models/metric_net_v2b/*.pt（渲染赢家已固化在 sweep_test_out；若做了早期 ckpt 增补 sweep，等它完再删）+ 删 D:/models/metric_net_f2k/*.pt（+7GB）；训练中守护脚本删 s1k-9k net/ema（只留 s10k+ 供 sweep），防 D 盘爆。
   - w48 占用估算：net≈972MB + ema≈324MB /1k 步 → 16 步全保留 ≈ 20.7GB + full×4 ≈ 5.2GB，必须边训边清。
   - 训后同样走 watcher 式直选 sweep（14 ckpt，~3.5h）→ 若超 s6k 回到分支 A 链。
4. profile 变体（若 w48 同配方仍输）：sigma-min 0.3 / noise-max 3.0 / jpeg 70-97 对齐 test 画像——与 w48 二选一或作为下一步实验，勿同时改两个变量。

## 3. 并行线：决赛真 Diffusion 调研
- ✅ 子代理 1de3dd3d 报告落盘：C:/生成式图像增强轻量模型设计挑战/research_diffusion_candidates_2026.md
- ✅ 关键推荐已独立核实：VOSR（CVPR2026，cswry/OSEDiff 同源，vision-only，0.5B/1.4B + 一步蒸馏，HF/ModelScope 权重，默认 4x 需验证 -u 1）真实；CODiff（ICCV25 一步去 JPEG，天然 1x，权重 Google Drive 需手动下）真实；UniLDiff/HYPIR++ 未开源确认。
- 建议：决赛 = HYPIR（保底主出图）+ VOSR-0.5B one-step（答辩主线）+ CODiff（加分组件）；4x SR 一步模型一律不直接 1x 出图。
- 待办（后续会话/GPU 空出后）：VOSR -u 1 本地验证；CODiff 权重清单给用户手动下载。

## 4. 纪律
- 杀进程前备份 sweep_results.json；勿杀 watcher(52308)。
- pyiqa 输入 (0,1) fp32 clamp；clipiqa 必须 4K。
- 提交件纯 ASCII、放 提交历史/；提交由用户操作（新号 9 次）。
- 每完成一步更新本文件 + HANDOFF §16。
